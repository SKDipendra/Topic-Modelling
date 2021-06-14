import pandas as pd
from pandas import  DataFrame
import numpy as np
import json
import csv
from transformers import BertTokenizer
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random
import time
import datetime
import plotly.express as px
from sklearn.metrics import matthews_corrcoef, confusion_matrix,accuracy_score,f1_score
from keras.preprocessing.sequence import pad_sequences


#using cpu as our device
device = torch.device("cpu")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load the dataset into a pandas dataframe.
df = pd.read_csv("spans_final.csv", header=None, names=['id','primary_frame','end','start','text'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

#with pd.option_context('display.max_rows', 1, 'display.max_columns', None):
    #print(df)


# Get the lists of sentences and their labels.
texts = df.text.values
primary_frames = df.primary_frame.values
primary_framess = primary_frames.astype(np.long)
print(primary_framess)

#showing the distribution of classes of primary frames
print('count', df.primary_frame.value_counts(dropna=False))

#below code measures the maximum text length
max_len = 0
avg_len = 0
input_ids_len = []



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#For every text..
for text in texts:
    #tokenize the text and add [CLS] and [SEP] tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids_len.append(len(input_ids))
    #Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

avg_len = np.mean(input_ids_len)

print('Max Text Length: ', max_len)
print('Average Text Length: ', avg_len)

plt.hist(input_ids_len, bins=100)
plt.show()

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []# For every sentence...
for text in texts:
    # `encode` will:
    #   (1) Tokenize the text.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

# Print text 0, now as a list of IDs.
#print('Original: ', texts[0])
#print('Token IDs:', input_ids[0])


# Set the maximum sequence length.
MAX_LEN = 128
print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")
print('\Done.')


# Create attention masks
attention_masks = []

# For each sentence...
for text in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in text]

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

attention_masks = np.array(attention_masks)

f1_score_list = []
from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=5) # Define the split - into 2 folds
for train_index, test_index in kf.split(df):
    train_inputs, validation_inputs = input_ids[train_index], input_ids[test_index]
    train_labels, validation_labels = primary_frames[train_index], primary_frames[test_index]

    train_masks, validation_masks = attention_masks[train_index], attention_masks[test_index]
    _, _ = primary_frames[train_index], primary_frames[test_index]

    df2 = pd.array(test_index)
    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs).to(torch.long)
    validation_inputs = torch.tensor(validation_inputs).to(torch.long)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 15, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model.cpu()

    optimizer = AdamW(model.parameters(),
                      lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 1# Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)



    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []



    # For each epoch...
    for epoch_i in range(0, epochs):
        predictions, true_labels = [], []

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.    print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')  # Measure how long the training epoch takes.
        t0 = time.time()  # Reset the total loss for this epoch.
        total_loss = 0  # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()  # For each batch of training data...
        for step, batch in enumerate(train_dataloader):  # Progress update every 40 batches.
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                            elapsed))  # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)  # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()  # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]  # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()  # Perform a backward pass to calculate the gradients.
            loss.backward()  # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           1.0)  # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()  # Update the learning rate.
            scheduler.step()  # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.    print("")
        print("Running Validation...")
        t0 = time.time()  # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()  # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0  # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():  # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]  # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            print('logits', logits)
            print('label_ids', label_ids)

            #append the predictions (max value index in logits) and true labelsn(label ids)
            predictions.extend((np.argmax(logits, axis=1).flatten()))
            print('prediction', predictions)
            true_labels.extend(label_ids.flatten())
            print('true labels', true_labels)

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy  # Track the number of batches
            nb_eval_steps += 1  # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print("")

        #here we check the f1 score
        F1 = f1_score(true_labels, predictions, average='macro')
        print('f1 score: ', F1)
    print("Training complete!")
    #for storing f1 score of every 3rd epoch to calculate the average f1 score for all the CV folds
    f1_score_list.append(F1)

avg_f1_score = sum(f1_score_list) / len(f1_score_list)
print('AVG f1:', avg_f1_score)



df2['prediction'] = predictions

#getting wrongly predicted rows
indices = [i for i in range(len(true_labels)) if true_labels[i] != predictions[i]]
wrong_predictions = df2.iloc[indices,:]
#wrong_predictions = list(wrong_predictions)
print('#################-W-R-O-N-G-###############################')
print(wrong_predictions.sample(30).to_string())



#getting rightly predicted rows
indices = [i for i in range(len(true_labels)) if true_labels[i] == predictions[i]]
right_predictions = df2.iloc[indices,:]
#right_predictions = list(right_predictions())
print('#######################-R-I-G-H-T-#########################')
print(right_predictions.sample(30).to_string())




def test_classifier():
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv("tobacco_test.csv", header=None, names=['id', 'primary_frame', 'text'])

    # report the number of sentences
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    df1 = df.dropna()
    print(df1.sample(5))

    # report the number of sentences
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    # Create sentence and label lists
    texts = df1.text.values
    primary_frames = df1.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []  # For every sentence...
    for text in texts:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)  # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long", truncating="post", padding="post")  # Create attention masks
    attention_masks = []  # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)  # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids).to(torch.long)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(primary_frames)  # Set the batch size.
    batch_size = 32  # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test setprint('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))# Put model in evaluation mode
    model.eval()  # Tracking variables
    predictions, true_labels = [], []  # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)



        # values prior to applying an activation function like the softmax.
        logits = outputs[0]  # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        print('logits', logits)
        print('label_ids', label_ids)

        # append the predictions (max value index in logits) and true labelsn(label ids)
        predictions.extend((np.argmax(logits, axis=1).flatten()))
        print('prediction', predictions)
        true_labels.extend(label_ids.flatten())
        print('true labels', true_labels)

    # here we check the f1 score
    F1 = f1_score(true_labels, predictions, average='macro')
    print('f1 score: ', F1)




