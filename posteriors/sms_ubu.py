import copy
from dataclasses import dataclass

import torch
import tqdm


@dataclass
class UBUConstants:
    h: torch.Tensor
    eta: torch.Tensor
    etam1g: torch.Tensor
    c11: torch.Tensor
    c21: torch.Tensor
    c22: torch.Tensor


def hper2const(h: torch.Tensor, gamma: torch.Tensor) -> UBUConstants:
    gh = gamma.double() * h.double()
    s = torch.sqrt(4 * torch.expm1(-gh / 2) - torch.expm1(-gh) + gh)
    eta = torch.exp(-gh / 2).float()
    etam1g = (-torch.expm1(-gh / 2) / gamma.double()).float()
    c11 = (s / gamma.double()).float()
    c21 = (torch.exp(-gh) * torch.expm1(gh / 2.0).pow(2) / s).float()
    c22 = (torch.sqrt(8 * torch.expm1(-gh / 2) - 4 * torch.expm1(-gh) - gh * torch.expm1(-gh)) / s).float()
    return UBUConstants(h=h.float(), eta=eta, etam1g=etam1g, c11=c11, c21=c21, c22=c22)


def _ind_create(batch_it: int, n_batches: int) -> int:
    mod_it = batch_it % (2 * n_batches)
    if mod_it <= n_batches - 1:
        return mod_it
    return 2 * n_batches - mod_it - 1


class SMSUBUClassification(object):
    """
    Symmetric minibatch sampling (SMS) + UBU kinetic Langevin sampler.

    This implementation follows the same practical structure used in the
    public SMS_Kinetic_Langevin notebooks while matching this repository's
    posterior API style.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_fn: torch.nn.Module,
        step_size: float = 1e-3,
        gamma: float = 1e-2,
        l2reg: float = 1e-4,
        l2reg_extra: float = 0.0,
        swa_epochs: int = 0,
        swa_lr: float = 1e-3,
        swa_weight_decay: float = 0.0,
        burnin_epochs: int = 5,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.compute_dtype = compute_dtype
        self.network = copy.deepcopy(network)
        self.device = next(self.network.parameters()).device
        self.network = self.network.to(self.device, dtype=self.compute_dtype)
        self.loss_fn = loss_fn
        self.step_size = float(step_size)
        self.gamma = float(max(gamma, 1e-12))
        self.l2reg = float(max(l2reg, 0.0))
        self.l2reg_extra = float(max(l2reg_extra, 0.0))
        self.swa_epochs = int(max(swa_epochs, 0))
        self.swa_lr = float(max(swa_lr, 0.0))
        self.swa_weight_decay = float(max(swa_weight_decay, 0.0))
        self.burnin_epochs = int(max(burnin_epochs, 0))

        self._samples = []

    def _materialize_batches(self, loader):
        # Keep cached batches on CPU and move only the current batch to GPU.
        batches = []
        for x, y in loader:
            batches.append((x.detach().cpu(), y.detach().cpu()))
        return batches

    def _get_batch(self, batches, batch_idx):
        x_cpu, y_cpu = batches[batch_idx]
        x = x_cpu.to(self.device, dtype=self.compute_dtype, non_blocking=True)
        y = y_cpu.to(self.device, non_blocking=True)
        return x, y

    def _set_grad_info(self, model):
        grad_info = []
        for name, p in model.named_parameters():
            grad_info.append({"is_bias": ("bias" in name)})
        return grad_info

    def _u_step(self, p, v, hc):
        xi1 = torch.randn_like(p)
        xi2 = torch.randn_like(p)
        pn = p + hc.etam1g * v + hc.c11 * xi1
        vn = v * hc.eta + hc.c21 * xi1 + hc.c22 * xi2
        return pn, vn

    def _half_u_step_pair(self, net1, net2, hc):
        with torch.no_grad():
            for p, q in zip(net1.parameters(), net2.parameters()):
                xi1 = torch.randn_like(p.data)
                xi2 = torch.randn_like(p.data)

                p.data = p.data + hc.etam1g * p.v + hc.c11 * xi1
                p.v = p.v * hc.eta + hc.c21 * xi1 + hc.c22 * xi2

                q.data = q.data + hc.etam1g * q.v + hc.c11 * xi1
                q.v = q.v * hc.eta + hc.c21 * xi1 + hc.c22 * xi2

    def _likelihood_grads(self, model, x, y):
        out = model(x)
        loss = self.loss_fn(out, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            bsz = y.shape[0]
            grads = [p.grad.detach().clone() * bsz for p in model.parameters()]
        return grads, loss.detach()

    def _reference_batch_grad(self, batch_idx, batches):
        x, y = self._get_batch(batches, batch_idx)
        grads, _ = self._likelihood_grads(self._net_star, x, y)
        return grads

    def _svrg_grad(self, model, batch_idx, batches):
        x, y = self._get_batch(batches, batch_idx)
        grads_lik, loss = self._likelihood_grads(model, x, y)
        grads_star_batch = self._reference_batch_grad(batch_idx, batches)

        with torch.no_grad():
            grads_svrg = []
            n_batches = len(batches)
            model_params = list(model.parameters())
            for i, (p, g_lik, p_star, g_star_batch, g_star_full) in enumerate(
                zip(
                    model_params,
                    grads_lik,
                    self._net_star.parameters(),
                    grads_star_batch,
                    self._net_star_full_grad,
                )
            ):
                g_reg = torch.zeros_like(p)
                if not self._param_info[i]["is_bias"] and self.l2reg > 0:
                    g_reg = self.l2reg * p.data

                g = g_reg + g_star_full + (g_lik - g_star_batch) * n_batches
                if self.l2reg_extra > 0:
                    g = g + self.l2reg_extra * (p.data - p_star.data)
                grads_svrg.append(g)

        return grads_svrg, loss

    def _prepare_reference_point(self, batches):
        self._net_star = copy.deepcopy(self.network)
        self._param_info = self._set_grad_info(self._net_star)
        self._net_star_full_grad = [torch.zeros_like(p) for p in self._net_star.parameters()]

        for batch_idx in range(len(batches)):
            x, y = self._get_batch(batches, batch_idx)
            grads, _ = self._likelihood_grads(self._net_star, x, y)
            with torch.no_grad():
                for g_full, g in zip(self._net_star_full_grad, grads):
                    g_full.add_(g)

    def _run_swa_warmup(self, batches, verbose=False, extra_verbose=False):
        if self.swa_epochs == 0 or self.swa_lr == 0.0:
            return

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.swa_lr,
            weight_decay=self.swa_weight_decay,
        )

        averaged_state = copy.deepcopy(self.network.state_dict())
        avg_count = 0

        if verbose:
            swa_epochs = tqdm.trange(self.swa_epochs, desc="swa")
        else:
            swa_epochs = range(self.swa_epochs)

        for _ in swa_epochs:
            epoch_loss = 0.0
            batch_order = torch.randperm(len(batches)).tolist()

            if extra_verbose:
                batch_iter = tqdm.tqdm(batch_order, leave=False)
            else:
                batch_iter = batch_order

            for batch_idx in batch_iter:
                x, y = self._get_batch(batches, batch_idx)
                loss = self.loss_fn(self.network(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            avg_count += 1
            current_state = self.network.state_dict()
            with torch.no_grad():
                for key, avg_tensor in averaged_state.items():
                    current_tensor = current_state[key]
                    if torch.is_floating_point(avg_tensor):
                        avg_tensor.mul_((avg_count - 1) / avg_count)
                        avg_tensor.add_(current_tensor, alpha=1.0 / avg_count)
                    else:
                        avg_tensor.copy_(current_tensor)

            if verbose:
                swa_epochs.set_postfix({"loss": f"{epoch_loss / len(batches):.4f}"})

        self.network.load_state_dict(averaged_state)

    def _ubu_step2(self, net, net2, hc, batch_idx_list, batches):
        self._half_u_step_pair(net, net2, hc)

        grads2, _ = self._svrg_grad(net2, batch_idx_list[0], batches)
        with torch.no_grad():
            for q, g in zip(net2.parameters(), grads2):
                q.v -= hc.h * g

        self._half_u_step_pair(net, net2, hc)

        grads, loss = self._svrg_grad(net, batch_idx_list[2], batches)
        with torch.no_grad():
            for p, g in zip(net.parameters(), grads):
                p.v -= 2.0 * hc.h * g

        self._half_u_step_pair(net, net2, hc)

        grads2b, _ = self._svrg_grad(net2, batch_idx_list[1], batches)
        with torch.no_grad():
            for q, g in zip(net2.parameters(), grads2b):
                q.v -= hc.h * g

        self._half_u_step_pair(net, net2, hc)

        return loss

    def train(self, loader, epochs=10, verbose=False, extra_verbose=False):
        batches = self._materialize_batches(loader)
        n_batches = len(batches)
        if n_batches == 0:
            raise RuntimeError("SMS-UBU received an empty training loader.")

        epochs = int(max(epochs, 1))

        self._run_swa_warmup(batches, verbose=verbose, extra_verbose=extra_verbose)

        self._prepare_reference_point(batches)

        net = copy.deepcopy(self._net_star)
        net2 = copy.deepcopy(self._net_star)

        with torch.no_grad():
            gamma_t = torch.tensor(self.gamma, device=self.device)
            h_t = torch.tensor(self.step_size / 2.0, device=self.device)
            hc = hper2const(h_t, gamma_t)

            for p, q in zip(net.parameters(), net2.parameters()):
                p.v = torch.randn_like(p)
                q.v = p.v.clone()

        self._samples = []
        last_loss = 0.0

        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        for epoch in pbar:
            epoch_loss = 0.0

            rperm = torch.randperm(n_batches)
            rperm2 = torch.randperm(n_batches)

            if extra_verbose:
                inner = tqdm.trange(n_batches, leave=False)
            else:
                inner = range(n_batches)

            for i in inner:
                it = epoch * n_batches + i
                ind = _ind_create(2 * it, n_batches)
                ind2 = _ind_create(2 * it + 1, n_batches)
                indc = _ind_create(it, n_batches)
                batch_idx_list = [int(rperm2[ind]), int(rperm2[ind2]), int(rperm[indc])]

                loss = self._ubu_step2(net, net2, hc, batch_idx_list, batches)
                epoch_loss += float(loss.item())

            last_loss = epoch_loss / n_batches

            collect = epoch >= self.burnin_epochs
            if collect:
                self._samples.append(copy.deepcopy(net.state_dict()))

            if verbose:
                pbar.set_postfix({"loss": f"{last_loss:.4f}", "samples": len(self._samples)})

        if len(self._samples) == 0:
            self._samples.append(copy.deepcopy(net.state_dict()))

        train_acc = self._accuracy_from_state(self._samples[-1], batches)
        return last_loss, train_acc

    def _accuracy_from_state(self, state_dict, batches):
        eval_net = copy.deepcopy(self.network)
        eval_net.load_state_dict(state_dict)
        eval_net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx in range(len(batches)):
                x, y = self._get_batch(batches, batch_idx)
                logits = eval_net(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
        return float(correct / max(total, 1))

    def test(self, loader, verbose=False):
        if len(self._samples) == 0:
            raise RuntimeError("SMS-UBU has no posterior samples. Run train(...) first.")

        eval_net = copy.deepcopy(self.network)
        eval_net.eval()

        all_sample_logits = []

        if verbose:
            sample_iter = tqdm.tqdm(self._samples)
        else:
            sample_iter = self._samples

        for state_dict in sample_iter:
            eval_net.load_state_dict(state_dict)
            batch_logits = []
            with torch.no_grad():
                for x, _ in loader:
                    batch_logits.append(eval_net(x.to(self.device, dtype=self.compute_dtype)).detach().cpu())
            all_sample_logits.append(torch.cat(batch_logits, dim=0))

        return torch.stack(all_sample_logits, dim=0)

    def UncertaintyPrediction(self, loader, verbose=False):
        logits = self.test(loader, verbose=verbose)
        probs = logits.softmax(-1)
        return probs.mean(0), probs.var(0)
