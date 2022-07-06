import json
import torch
import nltk
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from data_loader import get_loader, get_val_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.metrics import bleu_score


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if image.size != (256, 256):
        image = image.resize([256, 256], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def split_before_end(output):
    for i in range(len(output)):
        if int(output[i]) == 2:
            return output[1:i]


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_val_loader(args.image_dir, args.caption_path, vocab,
                                 transform, args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    # Build models
    # eval mode (batchnorm uses moving mean/variance)
    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(
        args.embed_size,
        args.hidden_size,
        len(vocab),
        args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    try:
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(args.encoder_path))
        else:
            encoder.load_state_dict(
                torch.load(
                    args.encoder_path,
                    map_location=torch.device('cpu')))
    except BaseException as e:
        print(e)
    try:
        if torch.cuda.is_available():
            decoder.load_state_dict(torch.load(args.decoder_path))
        else:
            decoder.load_state_dict(
                torch.load(
                    args.decoder_path,
                    map_location=torch.device('cpu')))
    except BaseException as e:
        print(e)

    total_bleu_score = 0
    count = 0
    img_bleu_score = {}
    for i, (images, captions, img_ids) in enumerate(data_loader):
        print(i)

        # Set mini-batch dataset
        images = images.to(device)
        # captions = captions.to(device)
        # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        with torch.no_grad():
            features = encoder(images)
            outputs = decoder.sample(features)
        tmp = 0
        for o, c, img_id in zip(outputs, captions, img_ids):
            o = split_before_end(o)
            c = [split_before_end(_c) for _c in c]
            if o is not None and c is not None:
                #o = [vocab.idx2word[int(_o)] for _o in o]
                #c = [vocab.idx2word[int(_c)] for _c in c]
                o = [str(int(_o)) for _o in o]
                c = [[str(int(__c)) for __c in _c] for _c in c]
                tmp_score = bleu_score([o], [c], 4)
                img_bleu_score[img_id] = tmp_score
                tmp += tmp_score
                count += 1
        total_bleu_score += tmp
        print(tmp)
    print(total_bleu_score / count)
    with open(f'bleu_score.txt', 'a') as f:
        f.write(f'512-3000: {total_bleu_score / count}\n')
    with open(f'img_bleu_score.json', 'w') as f:
        json.dump(img_bleu_score, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoder_path',
        type=str,
        default='models/encoder512-5-3000.ckpt',
        help='path for trained encoder')
    parser.add_argument(
        '--decoder_path',
        type=str,
        default='models/decoder512-5-3000.ckpt',
        help='path for trained decoder')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='data/vocab.pkl',
        help='path for vocabulary wrapper')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/resizedval2014',
        help='directory for resized images')
    parser.add_argument(
        '--caption_path',
        type=str,
        default='data/annotations/captions_val2014.json',
        help='path for train annotation json file')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument(
        '--embed_size',
        type=int,
        default=512,
        help='dimension of word embedding vectors')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='dimension of lstm hidden states')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
