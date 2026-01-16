from torch.utils.data import Dataset, DataLoader
import torch
import os
import tqdm
import io
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import numpy as np
import sklearn.metrics as sk

class MovieReviewsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, path, use_tokenizer):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop 
    # through each label.
    for label in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      # Go through each file and read its content.
      for file_name in tqdm.tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        # content = fix_text(content) @TODO: implement fix_text function.
        # Save content.
        self.texts.append(content)
        # Save encode labels.
        self.labels.append(label)

    # Number of exmaples.
    self.n_examples = len(self.labels)
    
    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """

    return {'text':self.texts[item],
            'label':self.labels[item]}
  
class Gpt2ClassificationCollator(object):
  r"""
  Data Collator used for GPT2 in a classification task. 
  
  It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
  can go straight into a GPT2 model.

  This class is built with reusability in mind: it can be used as is as long
  as the `dataloader` outputs a batch in dictionary format that can be passed 
  straight into the model - `model(**batch)`.

  Arguments:

    use_tokenizer (:obj:`transformers.tokenization_?`):
        Transformer type tokenizer used to process raw text into numbers.

    labels_ids (:obj:`dict`):
        Dictionary to encode any labels names into numbers. Keys map to 
        labels names and Values map to number associated to those labels.

    max_sequence_len (:obj:`int`, `optional`)
        Value to indicate the maximum desired sequence to truncate or pad text
        sequences. If no value is passed it will used maximum sequence size
        supported by the tokenizer and model.

  """

  def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

      # Tokenizer to be used inside the class.
      self.use_tokenizer = use_tokenizer
      # Check max sequence length.
      self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
      # Label encoder used inside the class.
      self.labels_encoder = labels_encoder

      return

  def __call__(self, sequences):
      r"""
      This function allowes the class objesct to be used as a function call.
      Sine the PyTorch DataLoader needs a collator function, I can use this 
      class as a function.

      Arguments:

        item (:obj:`list`):
            List of texts and labels.

      Returns:
        :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
        It holddes the statement `model(**Returned Dictionary)`.
      """

      # Get all texts from sequences list.
      texts = [sequence['text'] for sequence in sequences]
      # Get all labels from sequences list.
      labels = [sequence['label'] for sequence in sequences]
      # Encode all labels using label encoder.
      labels = torch.tensor([self.labels_encoder[label] for label in labels])
      # Call tokenizer on all texts to convert into tensors of numbers with 
      # appropriate padding.
      inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
      # print(inputs)
      # # Update the inputs with the associated encoded labels as tensor.
      # inputs.update({'labels':torch.tensor(labels)})

      return inputs, labels

class LLAClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classification task. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = torch.tensor([self.labels_encoder[label] for label in labels])
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # print(inputs)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs
    

class GPT2Functional(torch.nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, data) -> torch.Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits
    
class GPT2FeaturesFunctional(torch.nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model.transformer

    def forward(self, data) -> torch.Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.last_hidden_state[:,-1,:]
    
def train(model, dataloader, optimizer_, scheduler_, device_):
    total_loss = 0
    model.train()
    predictions_labels = []
    true_labels = []

    # For each batch of training data...
    for x, y in tqdm.tqdm(dataloader, total=len(dataloader)):

        true_labels += y.numpy().flatten().tolist()
        
        # move batch to device
        x = {k:v.type(torch.long).to(device_) for k,v in x.items()}

        model.zero_grad()
        outputs = model(input_ids = x['input_ids'],
                        attention_mask = x['attention_mask'],
                        labels = y.type(torch.long).to(device_))

        loss, logits = outputs[:2]
        loss.backward()

        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer_.step()
        scheduler_.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
    
    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def validation(model, dataloader, device_):
    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.eval()

    for x, y in tqdm.tqdm(dataloader, total=len(dataloader)):

        # Add original labels - use later for evaluation.
        true_labels += y.numpy().flatten().tolist()
        
        # move batch to device
        x = {k:v.type(torch.long).to(device_) for k,v in x.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            outputs = model(input_ids = x['input_ids'],
                        attention_mask = x['attention_mask'],
                        labels = y.type(torch.long).to(device_))

            loss, logits = outputs[:2]
            
            logits = logits.detach().cpu().numpy()

            total_loss += loss.item()
            
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def sort_preds_index(pi,yi):
    return (pi.argmax(1) == yi)

def aucroc(id_scores,ood_scores):
    '''
    INPUTS: scores should be maximum softmax probability for each test example
    '''
    labels = np.zeros((id_scores.shape[0] + ood_scores.shape[0]), dtype=np.int32)
    labels[:id_scores.shape[0]] += 1
    examples = np.squeeze(np.hstack((id_scores, ood_scores)))
    return sk.roc_auc_score(labels, examples)