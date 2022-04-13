from utils import EarlyStopping, LRScheduler, topk_correct, save_checkpoint
import torch
from torchvision import transforms
import time
import  numpy as np
import logging
import os
import gc
from matplotlib import pyplot as plt
import json

from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from models.Captioner import Encoder, DecoderWithAttention, Decoder_NoTFNoAttNoGlove
from data import CaptionDataset
from embedding import get_embedding_matrix

gc.collect()
torch.cuda.empty_cache()

# Data parameters
train_file = '/home/gaurangajitk/DL/data/image-caption-data/annotations_train1.txt'
val_file = '/home/gaurangajitk/DL/data/image-caption-data/annotations_val.txt'
data_folder = '/home/gaurangajitk/DL/data/image-caption-data'  # folder with data files saved by create_input_files.py
data_name = '5_cap_per_img_15_min_word_freq'  # base name shared by data files
bleu_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
checkpoint_path = None
model_type = ''

def train(dataloader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, device):
    loss_epoch = correct = total = 0
    decoder.train()
    if config['fine_tune_encoder']:
        encoder.train()
    for batch_idx, batch in enumerate(dataloader):
        imgs, caps, caplens = batch
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # Add doubly stochastic attention regularization
        loss += config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if config['fine_tune_encoder']:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        decoder_optimizer.step()
        if config['fine_tune_encoder']:
            encoder_optimizer.step()

        # metrics
        loss_epoch += loss.item()
        total += targets.data.size(0)
        correct += topk_correct(scores.data, targets.data, 5)
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

def test(dataloader, encoder, decoder, criterion, config, device):
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    loss_epoch = correct = total = 0
    decoder.eval()
    if config['fine_tune_encoder']:
        encoder.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs, caps, caplens, allcaps, meta = batch
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += config['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # metrics
            loss_epoch += loss.item()
            total += targets.data.size(0)
            correct += topk_correct(scores.data, targets.data, 5)

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)    #batch x max(seq_len). 1 final sentence prediction per sample
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    bleu1 = corpus_bleu(references, hypotheses, weights=[1,0,0,0])
    bleu2 = corpus_bleu(references, hypotheses, weights=[0,1,0,0])
    bleu3 = corpus_bleu(references, hypotheses, weights=[0,0,1,0])
    bleu4 = corpus_bleu(references, hypotheses, weights=[0,0,0,1])
    cum_BLEU = corpus_bleu(references, hypotheses)
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy, bleu1, bleu2, bleu3, bleu4, cum_BLEU


def main():
    start = time.time()
    np.random.seed(345)
    torch.manual_seed(345)
    logging.basicConfig(filename=f'./logs/training_{data_name}_{model_type}.log', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info('\n************************************\n')
    print('\n************************************\n')
    global word_map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    

    config = {
        'encoded_image_size': 14,
        'fine_tune_encoder': False,
        'attention_dim': 512,
        'embed_dim': 200,
        'decoder_dim': 512,
        'dropout': 0.5,
        'encoder_dim': 2048,
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-3,
        'batch_size': 128,
        'num_epochs': 50,
        'alpha_c': 1    # regularization parameter for 'doubly stochastic attention', as in the paper
    }

    print(config)
    logging.info(config)
    print('Preparing Datasets')
    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    #add random flip transform later
        ])
    
    trainset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=data_transforms)
    valset = CaptionDataset(data_folder, data_name, 'VAL', transform=data_transforms)
    testset = CaptionDataset(data_folder, data_name, 'TEST', transform=data_transforms)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    print('Setup metrics')
    logging.info('Setup metrics')

    best_valid_acc = best_valid_bleu = 0
    train_history_loss = []
    train_history_acc = []
    val_history_loss = []
    val_history_acc = []
    val_history_bleu1 = []
    val_history_bleu2 = []
    val_history_bleu3 = []
    val_history_bleu4 = []
    val_history_cumBleu = []

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    print('Is there a checkpoint?')
    if checkpoint_path:
        print('\t YES')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        start_epoch = checkpoint['epoch'] + 1
        best_valid_bleu = checkpoint['best_bleu']

        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    else:
        print('\t NO')
        start_epoch = 0

        print('Creare Model')
        encoder = Encoder(config)
        decoder = DecoderWithAttention(config, vocab_size=len(word_map))
        
        # print('Load pretrained glove embeddidngs')
        # word_embedding = get_embedding_matrix(word_map.keys())
        # decoder.load_pretrained_embeddings(word_embedding)
        # decoder.fine_tune_embeddings()

        print('Setup criterion and optimizer')
        criterion = torch.nn.CrossEntropyLoss()
        
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                lr=config['decoder_lr'])
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=config['encoder_lr']) if config['fine_tune_encoder'] else None
        encoder.to(device)
        decoder.to(device)

    lr_scheduler = LRScheduler(decoder_optimizer)
    early_stopping = EarlyStopping(patience=6, min_delta=1e-3)

    print('***** Training *****')
    logging.info('Started Training')

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start = time.time()
        train_loss, train_acc = train(train_loader, encoder, decoder, criterion, \
                                encoder_optimizer, decoder_optimizer, config, device)
        valid_loss, valid_acc, bleu1, bleu2, bleu3, bleu4, valid_bleu = test(val_loader, encoder, decoder, criterion, \
                                             config, device)
        time_for_epoch = time.time() - epoch_start

        print(f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, BLEU1= {bleu1:.3f}, BLEU2= {bleu2:.3f}, BLEU4= {bleu4:.3f},  CUM BLEU= {valid_bleu:.3f} \t Time Taken={time_for_epoch:.2f} s')
        logging.info(f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, BLEU1= {bleu1:.3f}, BLEU2= {bleu2:.3f}, BLEU4= {bleu4:.3f},  CUM BLEU= {valid_bleu:.3f} \t Time Taken={time_for_epoch:.2f} s')

        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            best_valid_acc = valid_acc
            state = {
                        'epoch': epoch, 
                        'encoder': encoder, 
                        'decoder': decoder,
                        'encoder_optimizer': encoder_optimizer,
                        'decoder_optimizer': decoder_optimizer,
                        'best_bleu': best_valid_bleu
                        }
            save_checkpoint(state, directory='./model_checkpoint', file_name=f'best_checkpoint_pt_{data_name}_{model_type}')
            logging.info(f'Checkpoint saved at Epoch {epoch}')
        

        lr_scheduler(valid_loss)
        early_stopping(-valid_bleu)
        #save losses for learning curves
        train_history_loss.append(train_loss)
        val_history_loss.append(valid_loss)
        train_history_acc.append(train_acc)
        val_history_acc.append(valid_acc)
        val_history_bleu1.append(bleu1)
        val_history_bleu2.append(bleu2)
        val_history_bleu3.append(bleu3)
        val_history_bleu4.append(bleu4)
        val_history_cumBleu.append(valid_bleu)
        if early_stopping.early_stop:
            break
    del encoder; del decoder; del encoder_optimizer; del decoder_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f'Final scheduler state {lr_scheduler.get_final_lr()}\n')

    print(f'Best test accuracy: {best_valid_acc}')
    print(f'Best test BLEU: {best_valid_bleu}')
    logging.info(f'Best test accuracy: {best_valid_acc}')
    logging.info(f'Best test BLEU: {best_valid_bleu}')

    # save curves
    plt.plot(range(len(train_history_loss)),train_history_loss, label="Training")
    plt.plot(range(len(val_history_loss)),val_history_loss, label="Validation")
    plt.legend()
    plt.title(f"Loss Curves: {model_type}")
    plt.savefig(f'curves/loss_curves_pt_{data_name}_{model_type}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    plt.plot(range(len(train_history_acc)),train_history_acc, label="Training Acc")
    plt.plot(range(len(val_history_acc)),val_history_acc, label="Validation Acc")
    plt.plot(range(len(val_history_bleu1)),val_history_bleu1, label="Validation BLEU-1")
    plt.plot(range(len(val_history_bleu2)),val_history_bleu2, label="Validation BLEU-2")
    plt.plot(range(len(val_history_bleu3)),val_history_bleu3, label="Validation BLEU-3")
    plt.plot(range(len(val_history_bleu4)),val_history_bleu4, label="Validation BLEU-4")
    plt.plot(range(len(val_history_cumBleu)),val_history_cumBleu, label="Validation BLEU (Cumulative)")
    plt.legend()
    plt.title(f"Accuracy Curves: {model_type}")
    plt.savefig(f'curves/acc_curves_pt_{data_name}_{model_type}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    # end
    diff = time.time() - start
    logging.info(f'Total time taken= {str(diff)} s')
    print(f'Total time taken= {str(diff)} s')


if __name__ == '__main__':
    main()