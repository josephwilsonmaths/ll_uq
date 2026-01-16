from torch.utils.data import DataLoader
import torch
import os
import tqdm
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from utils.helper_functions import *
import argparse
import time
from laplace import Laplace

parser = argparse.ArgumentParser(description='llm script')
parser.add_argument('--load_model', action='store_true', help='verbose flag for all methods')
parser.add_argument('--save_model', action='store_true', help='verbose flag for all methods')
parser.add_argument('--num_repeats', type=int, default=1, help='number of repeats for experiments')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and evaluation')
parser.add_argument('--max_length', type=int, default=None, help='maximum sequence length for tokenizer')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--load_posterior', action='store_true', help='flag to load trained posterior')
parser.add_argument('--posterior_epochs', type=int, default=2, help='number of LinearSampling training epochs')
parser.add_argument('--S', type=int, default=10, help='number of LinearSampling samples')
parser.add_argument('--lla', action='store_true', help='flag to run LLA method')
parser.add_argument('--dnn', action='store_true', help='flag to run DNN-GLM method')
parser.add_argument('--ll', action='store_true', help='flag to run LL-GLM method')
args = parser.parse_args()
    

#### _________________________ Setup _________________________ ####
if args.num_repeats == 1:
    set_seed(123)

max_length = args.max_length
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
labels_ids = {'neg': 0, 'pos': 1}
n_labels = len(labels_ids)

# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
model_config.num_labels = n_labels

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token

if args.num_repeats > 1:
    print(f'Running experiment for {args.num_repeats} repeats...')
    if args.dnn:
        ## DNN-GLM results
        dnn_glm_varroc_id_results = []
        dnn_glm_time_results = []
        dnn_glm_mem_results = []
    if args.ll:
        ## LL-GLM results   
        ll_glm_varroc_id_results = []
        ll_glm_time_results = []
        ll_glm_mem_results = []
    if args.lla:
        ## LLA results
        lla_varroc_id_results = []
        lla_time_results = []
        lla_mem_results = []

for repeat in range(args.num_repeats):
    print(f'----- Repeat {repeat+1} / {args.num_repeats} -----')

    #### _________________________ Prepare Data _________________________ ####
    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                            labels_encoder=labels_ids, 
                                                            max_sequence_len=max_length)

    lla_classification_collator = LLAClassificationCollator(use_tokenizer=tokenizer, 
                                                            labels_encoder=labels_ids, 
                                                            max_sequence_len=max_length)

    batch_size = args.batch_size

    # Create pytorch dataset.
    train_dataset = MovieReviewsDataset(path='data/imdb/aclImdb/train', 
                                use_tokenizer=tokenizer)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classification_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    print()

    valid_dataset =  MovieReviewsDataset(path='data/imdb/aclImdb/test', 
                                use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classification_collator)
    print('Created `valid_dataloader` with %d batches!'%len(valid_dataloader))

    print('Dealing with Train (LLA)...')
    # Create pytorch dataset.
    lla_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lla_classification_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))


    #### _________________________ Load Model _________________________ ####
    train_model = False
    if args.load_model and (args.num_repeats == 1):
        try:
            model_path = 'saved_models/gpt2_imdb_classification.pt'
            print('Loading model from `%s`'%model_path)
            model_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(model_dict['model_state_dict'])
            print(f'Model loaded!')
            print(f'Training details - Epochs: {model_dict["epochs"]}, Batch Size: {model_dict["batch_size"]}, Max Length: {model_dict["max_length"]}')
            print(f'Training Loss: {model_dict["train_loss"]:.5f}, Training Acc: {model_dict["train_acc"]:.5f}')
            print(f'Validation Loss: {model_dict["val_loss"]:.5f}, Validation Acc: {model_dict["val_acc"]:.5f}')
        except Exception as e:
            print('Error loading model from `%s`'%model_path)
            print(e)
            print('Continuing with randomly initialized model...')
            train_model = True
    else:
        train_model = True

    #### _________________________ Train Model _________________________ ####
    if train_model:
        print('Starting training...GPU available: %s'%torch.cuda.is_available())
        print(f'training for {args.epochs} epochs with batch size {batch_size} and max length {max_length}')
        epochs = args.epochs
        from sklearn.metrics import classification_report, accuracy_score
        from ml_things import fix_text

        optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = total_steps)

        for epoch in tqdm.tqdm(range(epochs)):
            # Perform one full pass over the training set.
            train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
            train_acc = accuracy_score(train_labels, train_predict)

            # Get prediction form model on validation data. 
            print('Validation on batches...')
            valid_labels, valid_predict, val_loss = validation(model, valid_dataloader, device)
            val_acc = accuracy_score(valid_labels, valid_predict)

            # Print loss and accuracy values to see how training evolves.
            print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
            print()

        # Save the trained model.
        if args.save_model and (args.num_repeats == 1):
            model_path = 'saved_models/gpt2_imdb_classification.pt'
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            print('Saving model to `%s`'%model_path)
            model_dict = {'model_state_dict': model.state_dict(),
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'max_length': max_length,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc}
            torch.save(model_dict, model_path)
            print('Model saved!')

    #### _________________________ Posterior Inference _________________________ ####

    import LinearSampling

    print('Starting posterior inference...')

    model_functional = GPT2Functional(model)
    model_features = GPT2FeaturesFunctional(model)

    if args.dnn:
        ###______________________ DNN-GLM Inference ________________________ ####
        print('Starting DNN-GLM inference...')

        # Timing start
        t0 = time.time()

        # Reset CUDA memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        dnn_glm = LinearSampling.Posteriors.Posterior(network=model_functional,
                                                    glm_type='DNN',
                                                    task = 'classification',
                                                    precision='single',
                                                    feature_extractor=model_features,
                                                    num_features=768,
                                                    num_outputs=2
                                                    )

        model_path = 'saved_models/dnn_glm.pt'
        train_posterior = False
        try:
            if args.load_posterior:
                print('Loading trained DNN-GLM posterior from `%s`'%model_path)
                dnn_glm.load_weights(model_path)
                print('Trained DNN-GLM posterior loaded!')
            else:
                train_posterior = True
        except Exception as e:
            print('Error loading trained DNN-GLM posterior from `%s`'%model_path)
            print(e)
            print('Continuing to train DNN-GLM posterior...')
            train_posterior = True

        if train_posterior:
            dnn_glm.train(train=train_dataset, 
                                bs=args.batch_size // 2, 
                                S=args.S,
                                gamma=0.01, 
                                lr=1e-5, 
                                epochs=args.posterior_epochs, 
                                mu=0.9,
                                verbose=True,
                                extra_verbose=True,
                                collate_fn=gpt2_classification_collator,
                                save_weights=model_path,
                                plot_loss_dir='testing')
                            
        ###______________________ DNN-GLM VARROC-ID EVAL ________________________ ####
        id_mean, id_var = dnn_glm.UncertaintyPrediction(test=valid_dataset, bs=args.batch_size // 2, network_mean=True, collate_fn=gpt2_classification_collator, verbose=True)

        test_targets = torch.cat([y for _,y in valid_dataloader])
        index_correct = sort_preds_index(id_mean,test_targets)

        max_index_correct = id_mean[index_correct,:].argmax(1)
        max_index_incorrect = id_mean[~index_correct,:].argmax(1)
        correct_var = id_var[index_correct,:]
        incorrect_var = id_var[~index_correct,:]
        pi_correct_var = correct_var[range(len(max_index_correct)),max_index_correct]
        pi_incorrect_var = incorrect_var[range(len(max_index_incorrect)),max_index_incorrect]

        var_roc_id = aucroc(pi_incorrect_var, pi_correct_var)

        t1 = time.time()
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0

        print('DNN-GLM VARROC-ID: %.5f'%var_roc_id)
        print('DNN-GLM inference time: %.2f seconds'%(t1-t0))
        print(f'DNN-GLM inference max memory: {mem:.2f} GB')

        if args.num_repeats > 1:
            ## DNN-GLM results
            dnn_glm_varroc_id_results.append(var_roc_id)
            dnn_glm_time_results.append(t1 - t0)
            dnn_glm_mem_results.append(mem)


    if args.ll:
        ###______________________ LL-GLM Inferenence ________________________ ####
        print('Starting LL-GLM inference...')

        # Timing start
        t0 = time.time()

        # Reset CUDA memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        ll_glm = LinearSampling.Posteriors.Posterior(network=model_functional,
                                                    glm_type='LL',
                                                    task = 'classification',
                                                    precision='single',
                                                    feature_extractor=model_features,
                                                    num_features=768,
                                                    num_outputs=2
                                                    )

        model_path = 'saved_models/ll_glm.pt'
        train_posterior = False
        try:
            if args.load_posterior:
                print('Loading trained LL-GLM posterior from `%s`'%model_path)
                ll_glm.load_weights(model_path)
                print('Trained LL-GLM posterior loaded!')
            else:
                train_posterior = True
        except Exception as e:
            print('Error loading trained LL-GLM posterior from `%s`'%model_path)
            print(e)
            print('Continuing to train LL-GLM posterior...')
            train_posterior = True

        if train_posterior:
            ll_glm.train(train=train_dataset, 
                                bs=args.batch_size, 
                                S=args.S,
                                gamma=0.01, 
                                lr=1e-5, 
                                epochs=args.posterior_epochs, 
                                mu=0.9,
                                verbose=True,
                                extra_verbose=True,
                                collate_fn=gpt2_classification_collator,
                                plot_loss_dir='testing')
                            

        ###______________________ LL-GLM VARROC-ID EVAL ________________________ ####
        id_mean, id_var = ll_glm.UncertaintyPrediction(test=valid_dataset, bs=args.batch_size, network_mean=True, collate_fn=gpt2_classification_collator, verbose=True)

        test_targets = torch.cat([y for _,y in valid_dataloader])
        index_correct = sort_preds_index(id_mean,test_targets)

        max_index_correct = id_mean[index_correct,:].argmax(1)
        max_index_incorrect = id_mean[~index_correct,:].argmax(1)
        correct_var = id_var[index_correct,:]
        incorrect_var = id_var[~index_correct,:]
        pi_correct_var = correct_var[range(len(max_index_correct)),max_index_correct]
        pi_incorrect_var = incorrect_var[range(len(max_index_incorrect)),max_index_incorrect]

        var_roc_id = aucroc(pi_incorrect_var, pi_correct_var)
        t1 = time.time()
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0

        print('LL-GLM VARROC-ID: %.5f'%var_roc_id)
        print('LL-GLM inference time: %.2f seconds'%(t1-t0))
        print(f'LL-GLM inference max memory: {mem:.2f} GB')

        if args.num_repeats > 1:
            ## LL-GLM results
            ll_glm_varroc_id_results.append(var_roc_id)
            ll_glm_time_results.append(t1 - t0)
            ll_glm_mem_results.append(mem)

    if args.lla:
        #### __________________________ LLA __________________________ ####
        # LLA Test
        print('Starting LLA inference...')

        # Timing start
        t0 = time.time()

        # Reset CUDA memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()    

        la = Laplace(
            model_functional,
            likelihood="classification",
            subset_of_weights="last_layer",
            hessian_structure="full",
            # This must reflect faithfully the reduction technique used in the model
            # Otherwise, correctness is not guaranteed
            feature_reduction="pick_last",
        )
        la.fit(lla_train_dataloader, progress_bar=True)
        la.optimize_prior_precision()

        print('LLA Trained')

        #______________________ LLA VARROC-ID EVAL ________________________ ####
        T = 10
        samples = []
        for x,_ in valid_dataloader:
            samples.append(la.predictive_samples(x=x,pred_type='glm',n_samples=T)) # output is (samples x batchsize x classes)
        samples = torch.cat(samples,dim=1).cpu()  # (samples,N,classes)
        id_mean, id_var = samples.mean(0), samples.var(0)  # (N,classes), (N,classes)

        test_targets = torch.cat([y for _,y in valid_dataloader])
        index_correct = sort_preds_index(id_mean,test_targets)

        max_index_correct = id_mean[index_correct,:].argmax(1)
        max_index_incorrect = id_mean[~index_correct,:].argmax(1)
        correct_var = id_var[index_correct,:]
        incorrect_var = id_var[~index_correct,:]
        pi_correct_var = correct_var[range(len(max_index_correct)),max_index_correct]
        pi_incorrect_var = incorrect_var[range(len(max_index_incorrect)),max_index_incorrect]

        var_roc_id = aucroc(pi_incorrect_var, pi_correct_var)
        t1 = time.time()
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0

        print('LLA VARROC-ID: %.5f'%var_roc_id)
        print('LLA inference time: %.2f seconds'%(t1-t0))
        print(f'LLA inference max memory: {mem:.2f} GB')

        # Store results if running multiple repeats
        if args.num_repeats > 1:
            ## LLA results
            lla_varroc_id_results.append(var_roc_id)
            lla_time_results.append(t1 - t0)
            lla_mem_results.append(mem)

if args.num_repeats > 1:
    # Final results over repeats
    print('----- Final Results over %d repeats -----'%args.num_repeats)
    if args.dnn:
        print('DNN-GLM VARROC-ID: %.5f +- %.5f'%(np.mean(dnn_glm_varroc_id_results), np.std(dnn_glm_varroc_id_results)))
        print('DNN-GLM Time: %.2f +- %.2f seconds'%(np.mean(dnn_glm_time_results), np.std(dnn_glm_time_results)))
        print('DNN-GLM Memory: %.2f +- %.2f GB'%(np.mean(dnn_glm_mem_results), np.std(dnn_glm_mem_results)))
    if args.ll:
        print('LL-GLM VARROC-ID: %.5f +- %.5f'%(np.mean(ll_glm_varroc_id_results), np.std(ll_glm_varroc_id_results)))
        print('LL-GLM Time: %.2f +- %.2f seconds'%(np.mean(ll_glm_time_results), np.std(ll_glm_time_results)))
        print('LL-GLM Memory: %.2f +- %.2f GB'%(np.mean(ll_glm_mem_results), np.std(ll_glm_mem_results)))
    if args.lla:
        print('LLA VARROC-ID: %.5f +- %.5f'%(np.mean(lla_varroc_id_results), np.std(lla_varroc_id_results)))
        print('LLA Time: %.2f +- %.2f seconds'%(np.mean(lla_time_results), np.std(lla_time_results)))
        print('LLA Memory: %.2f +- %.2f GB'%(np.mean(lla_mem_results), np.std(lla_mem_results)))
        
