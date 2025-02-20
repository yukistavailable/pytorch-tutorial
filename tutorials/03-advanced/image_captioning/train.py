import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, EncoderCNNWithAttention, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

fine_tune_encoder = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    if args.with_attention:
        # Build the models
        encoder = EncoderCNNWithAttention(args.pixel_num).to(device)
        encoder.fine_tune(fine_tune_encoder)
        decoder = DecoderRNNWithAttention(
            args.embed_size,
            args.hidden_size,
            len(vocab),
            args.num_layers, args.encoder_size, device).to(device)

        # Load the trained model parameters
        try:
            if args.encoder_path is not None:
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
            if args.decoder_path is not None:
                if torch.cuda.is_available():
                    decoder.load_state_dict(torch.load(args.decoder_path))
                else:
                    decoder.load_state_dict(
                        torch.load(
                            args.decoder_path,
                            map_location=torch.device('cpu')))
        except BaseException as e:
            print(e)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if fine_tune_encoder:
            params = list(decoder.parameters())
        else:
            params = list(decoder.parameters()) + \
                list(encoder.linear.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        encoder_optimizer = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad,
                encoder.parameters()),
            lr=args.learning_rate) if fine_tune_encoder else None
        alpha_c = args.alpha_c

        # Train the models
        total_step = len(data_loader)
        for epoch in range(args.num_epochs):
            for i, (images, captions, lengths) in enumerate(data_loader):

                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                # targets = pack_padded_sequence(
                #     captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = encoder(images)
                outputs, sorted_captions, alphas, sort_ind, decode_length = decoder(
                    features, captions, lengths)
                outputs = pack_padded_sequence(
                    outputs, decode_length, batch_first=True)[0]
                targets = sorted_captions[:, 1:]
                targets = pack_padded_sequence(
                    targets, decode_length, batch_first=True)[0]
                loss = criterion(outputs, targets)
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                decoder.zero_grad()
                encoder.zero_grad()
                optimizer.zero_grad()
                if fine_tune_encoder:
                    encoder_optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                if fine_tune_encoder:
                    encoder_optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' .format(
                        epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
                if (i + 1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), os.path.join(
                        args.model_path, 'decoder-with-attention-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                    torch.save(encoder.state_dict(), os.path.join(
                        args.model_path, 'encoder-with-attention-{}-{}.ckpt'.format(epoch + 1, i + 1)))
    else:
        # Build the models
        encoder = EncoderCNN(args.embed_size).to(device)
        encoder.fine_tune(fine_tune_encoder)
        decoder = DecoderRNN(
            args.embed_size,
            args.hidden_size,
            len(vocab),
            args.num_layers).to(device)

        # Load the trained model parameters
        try:
            if args.encoder_path is not None:
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
            if args.decoder_path is not None:
                if torch.cuda.is_available():
                    decoder.load_state_dict(torch.load(args.decoder_path))
                else:
                    decoder.load_state_dict(
                        torch.load(
                            args.decoder_path,
                            map_location=torch.device('cpu')))
        except BaseException as e:
            print(e)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if fine_tune_encoder:
            params = list(decoder.parameters())
        else:
            params = list(decoder.parameters()) + \
                list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        encoder_optimizer = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad,
                encoder.parameters()),
            lr=args.learning_rate) if fine_tune_encoder else None

        # Train the models
        total_step = len(data_loader)
        for epoch in range(args.num_epochs):
            for i, (images, captions, lengths) in enumerate(data_loader):

                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(
                    captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                decoder.zero_grad()
                encoder.zero_grad()
                optimizer.zero_grad()
                if fine_tune_encoder:
                    encoder_optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                if fine_tune_encoder:
                    encoder_optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' .format(
                        epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
                if (i + 1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), os.path.join(
                        args.model_path, 'decoder512-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                    torch.save(encoder.state_dict(), os.path.join(
                        args.model_path, 'encoder512-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoder_path',
        type=str,
        default=None,
        help='path for trained encoder')
    parser.add_argument(
        '--decoder_path',
        type=str,
        default=None,
        help='path for trained decoder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/',
        help='path for saving trained models')
    parser.add_argument(
        '--crop_size',
        type=int,
        default=224,
        help='size for randomly cropping images')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='data/vocab.pkl',
        help='path for vocabulary wrapper')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/resized2014',
        help='directory for resized images')
    parser.add_argument(
        '--caption_path',
        type=str,
        default='data/annotations/captions_train2014.json',
        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument(
        '--save_step',
        type=int,
        default=1000,
        help='step size for saving trained models')

    # Model parameters
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
    parser.add_argument('--pixel_num', type=int, default=16)
    parser.add_argument('--encoder_size', type=int, default=512)
    parser.add_argument('--alpha_c', type=float, default=1.0)

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--with_attention', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
