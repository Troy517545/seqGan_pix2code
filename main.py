from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

from utils.tree import Tree, tree_similarity
from utils.transforms import WordEmbedding, TreeToTensor, Vec2Word
from torchvision import transforms
from utils.transforms import Rescale, WordEmbedding, TreeToTensor, Vec2Word
from dataset import Pix2TreeDataset
import os


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 100
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 20
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64


pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

gen_path = './gen_EMBDIM32_HIDDENDIM32_MAXSEQLEN100.trc'
dis_path = './dis_EMBDIM64_HIDDENDIM64_MAXSEQLEN100.trc'

def samples_preprocessing():
    def count_word_dict(dataset):
        word_count = {'root':0, 'end':0}
        def count_tree(tree, word_count):
            for child in tree.children:
                count_tree(child, word_count)
            if tree.value in word_count:
                word_count[tree.value] += 1
            else:
                word_count[tree.value] = 1
        
        for i in range(len(dataset)):
            count_tree(dataset[i]['tree'], word_count)
        
        word_dict = {}
        i = 0
        for key in word_count.keys():
            a = np.zeros(len(word_count))
            a[i] = 1.0
            word_dict[key] = a
            i += 1
        return word_dict

    dataset = Pix2TreeDataset()
    if not os.path.exists('word_dict.npy'):
        word_dict = count_word_dict(dataset)
        np.save('word_dict.npy', word_dict)
    else:
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        
    # prepare dataset
    sample_dataset = Pix2TreeDataset(partition=range(int(len(dataset)*0.8)),
            tree_transform=transforms.Compose([WordEmbedding(word_dict),
                                               TreeToTensor()]),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))
    
    
    Vec2Word_t = Vec2Word(word_dict)

    sample_data = []
    
    for i in range(len(sample_dataset)):
        seq = Vec2Word_t(sample_dataset[i]['tree']).seq()
        seq = list(map(int, seq.split(' ')))
        seq = [x+1 for x in seq]
        if len(seq)>100:
            continue
        else:
            for i in range(100 - len(seq)):
                seq.append(0)
            sample_data.append(seq)

    return sample_data

def index_seq2word(seq):
    def count_word_dict(dataset):
        word_count = {'root':0, 'end':0}
        def count_tree(tree, word_count):
            for child in tree.children:
                count_tree(child, word_count)
            if tree.value in word_count:
                word_count[tree.value] += 1
            else:
                word_count[tree.value] = 1
        
        for i in range(len(dataset)):
            count_tree(dataset[i]['tree'], word_count)
        
        word_dict = {}
        i = 0
        for key in word_count.keys():
            a = np.zeros(len(word_count))
            a[i] = 1.0
            word_dict[key] = a
            i += 1
        return word_dict

    dataset = Pix2TreeDataset()
    if not os.path.exists('word_dict.npy'):
        word_dict = count_word_dict(dataset)
        np.save('word_dict.npy', word_dict)
    else:
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()

    index_dict = {}
    for k, v in word_dict.items():
        v_index = -1
        for v_index in range(14):
            if v[v_index] == 1:
                break
        index_dict[v_index] = k
    

    seq = seq.tolist()
    zero_index = 0
    for zero_index in range(100):
        if seq[zero_index] == 0:
            break
    if zero_index != 100:
        seq = seq[0:zero_index]

    seq = [x-1 for x in seq]
    seq = [index_dict[x] for x in seq]
    print(seq)
    return seq

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN


        print(' average_train_NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    losses = []

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()
        losses.append(pg_loss)

    print(' PG loss = %.4f' % (sum(losses) / len(losses)))

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training
    pos_val = real_data_samples[0:100]
    neg_val = generator.sample(100)

    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':

    # samples = samples_preprocessing()
    # samples = torch.LongTensor(samples)
    # torch.save(samples, 'samples.trc')
    samples = torch.load("samples.trc").type(torch.LongTensor)

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()
        samples = samples.cuda()

    POS_NEG_SAMPLES = samples.size()[0]
    # GENERATOR MLE TRAINING
    # print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    # train_generator_MLE(gen, gen_optimizer, samples, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # print("Saved generator model")

    gen.load_state_dict(torch.load(gen_path))
    print("Gererator model loaded")

    # PRETRAIN DISCRIMINATOR
    # print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    # train_discriminator(dis, dis_optimizer, samples, gen, d_steps = 50, epochs = 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # print("Saved discriminator model")

    dis.load_state_dict(torch.load(dis_path))
    print("Discriminator model loaded")

    # ADVERSARIAL TRAINING
    # print('\nStarting Adversarial Training...')


    # for epoch in range(ADV_TRAIN_EPOCHS):
    #     print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    #     # TRAIN GENERATOR
    #     print('\nAdversarial Training Generator : ', end='')
    #     sys.stdout.flush()
    #     train_generator_PG(gen, gen_optimizer, dis, 1)

    #     # TRAIN DISCRIMINATOR
    #     print('\nAdversarial Training Discriminator : ')
    #     train_discriminator(dis, dis_optimizer, samples, gen, d_steps = 5, epochs = 3)


    # torch.save(gen.state_dict(), gen_path)
    # print("Saved generator model")
    # torch.save(dis.state_dict(), dis_path)
    # print("Saved discriminator model")

    generated_samples = gen.sample(5)

    seq = index_seq2word(generated_samples[0])