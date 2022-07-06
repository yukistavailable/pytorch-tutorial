import torch
import matplotlib.pyplot as plt
import numpy as p
import argparse
import pickle
import os
import requests
import glob
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if image.size != (256, 256):
        image = image.resize([256, 256], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def generate_sentence(
        image_path,
        encoder_path,
        decoder_path,
        vocab_path,
        embed_size,
        hidden_size,
        num_layers):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    # eval mode (batchnorm uses moving mean/variance)
    encoder = EncoderCNN(embed_size).eval()
    decoder = DecoderRNN(
        embed_size,
        hidden_size,
        len(vocab),
        num_layers).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    try:
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(encoder_path))
        else:
            encoder.load_state_dict(
                torch.load(
                    encoder_path,
                    map_location=torch.device('cpu')))
    except BaseException as e:
        print(e)
    try:
        if torch.cuda.is_available():
            decoder.load_state_dict(torch.load(decoder_path))
        else:
            decoder.load_state_dict(
                torch.load(
                    decoder_path,
                    map_location=torch.device('cpu')))
    except BaseException as e:
        print(e)

    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    with torch.no_grad():
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
    # (1, max_seq_length) -> (max_seq_length)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence


def generate_sentence_from_image_url(
        image_url,
        encoder_path,
        decoder_path,
        vocab_path,
        embed_size,
        hidden_size,
        num_layers):
    response = requests.get(image_url)
    filename = os.path.basename(image_url)
    files = glob.glob0('data/images', filename)
    if len(files) == 0:
        print('download image ...')
        if response.status_code == 200:
            image = response.content
            with open(os.path.join('data/images', filename), 'wb') as f:
                f.write(image)
            print('downloading is success')
        else:
            print('error')
    image_path = os.path.join('data/images', filename)
    return generate_sentence(
        image_path,
        encoder_path,
        decoder_path,
        vocab_path,
        embed_size,
        hidden_size,
        num_layers)


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

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

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    with torch.no_grad():
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
    # (1, max_seq_length) -> (max_seq_length)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument(
        '--encoder_path',
        type=str,
        default='models/encoder512-2-3000.ckpt',
        help='path for trained encoder')
    parser.add_argument(
        '--decoder_path',
        type=str,
        default='models/decoder512-2-3000.ckpt',
        help='path for trained decoder')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='data/vocab.pkl',
        help='path for vocabulary wrapper')

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
