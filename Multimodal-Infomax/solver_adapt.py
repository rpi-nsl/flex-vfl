import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model_top import MMIM_top
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet

def make_optimizers(model, text_enc, acoustic_enc, visual_enc, hp, epoch):
    # Create optimizers and schedulers
    optimizer={}
    optimizer_visual={}
    optimizer_acoustic={}

    mmilb_param = []
    main_param = []
    bert_param = []
    visual_param = []
    acoustic_param = []

    decay_factor = 1
    #if epoch > 25:
    #    decay_factor = 0.5

    # make list of parameters
    for name, p in text_enc.named_parameters():
        # print(name)
        if p.requires_grad:
            if 'bert' in name:
                bert_param.append(p)

    for name, p in acoustic_enc.named_parameters():
        # print(name)
        if p.requires_grad:
            #main_param.append(p)
            acoustic_param.append(p)

    for name, p in visual_enc.named_parameters():
        # print(name)
        if p.requires_grad:
            #main_param.append(p)
            visual_param.append(p)

    for name, p in model.named_parameters():
        # print(name)
        if p.requires_grad:
            if 'mi' in name:
                mmilb_param.append(p)
            else: 
                main_param.append(p)
        
        for p in (mmilb_param+main_param):
            if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                nn.init.xavier_normal_(p)

    # Create optimizers
    optimizer_mmilb = getattr(torch.optim, hp.optim)(
        mmilb_param, lr=hp.lr_mmilb*decay_factor, weight_decay=hp.weight_decay_club)
    
    optimizer_main_group = [
        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main*decay_factor, 'momentum': hp.momentum}
    ]

    optimizer_main = getattr(torch.optim, hp.optim)(
        optimizer_main_group
    )
    
    optimizer_text_group = [
        {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert*decay_factor, 'momentum': hp.momentum},
    ]

    optimizer_text = getattr(torch.optim, hp.optim)(
        optimizer_text_group
    )
    
    optimizer_visual_group = [
        {'params': visual_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main*decay_factor, 'momentum': hp.momentum}
    ]

    optimizer_visual = getattr(torch.optim, hp.optim)(
        optimizer_visual_group
    )
    
    optimizer_acoustic_group = [
        {'params': acoustic_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main*decay_factor, 'momentum': hp.momentum}
    ]

    optimizer_acoustic = getattr(torch.optim, hp.optim)(
        optimizer_acoustic_group
    )

    # Create schedulers for learning rate changes
    scheduler_mmilb = ReduceLROnPlateau(optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
    scheduler_main = ReduceLROnPlateau(optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)
    scheduler_text = ReduceLROnPlateau(optimizer_text, mode='min', patience=hp.when, factor=0.5, verbose=True)
    scheduler_acoustic = ReduceLROnPlateau(optimizer_acoustic, mode='min', patience=hp.when, factor=0.5, verbose=True)
    scheduler_visual = ReduceLROnPlateau(optimizer_visual, mode='min', patience=hp.when, factor=0.5, verbose=True)

    return (optimizer_main, optimizer_mmilb, optimizer_text, optimizer_acoustic, optimizer_visual,
    scheduler_mmilb, scheduler_main, scheduler_text, scheduler_acoustic, scheduler_visual)


class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.alg = hp.alg
        self.scores = []
        self.start_epoch = 1
        self.server_time = hp.server_time
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.suffix = f'_alg{self.alg}_servertime_{self.server_time}_dataset{hp.dataset}_lrmain{hp.lr_main}_lrbert{hp.lr_bert}_decay{hp.weight_decay_main}_seed{hp.seed}'
        #self.suffix = f'_alg{self.alg}_servertime_{self.server_time}_dataset{hp.dataset}_seed{hp.seed}'

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.text_enc = text_enc = LanguageEmbeddingLayer(hp)
            self.visual_enc = visual_enc = RNNEncoder(
                in_size = hp.d_vin,
                hidden_size = hp.d_vh,
                out_size = hp.d_vout,
                num_layers = hp.n_layer,
                dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
                bidirectional = hp.bidirectional
            )
            self.acoustic_enc = acoustic_enc = RNNEncoder(
                in_size = hp.d_ain,
                hidden_size = hp.d_ah,
                out_size = hp.d_aout,
                num_layers = hp.n_layer,
                dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
                bidirectional = hp.bidirectional
            )
            self.model = model = MMIM_top(hp)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            text_enc = text_enc.cuda()
            acoustic_enc = acoustic_enc.cuda()
            visual_enc = visual_enc.cuda()
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")
        
        # optimizer
        (self.optimizer_main, self.optimizer_mmilb, self.optimizer_text, 
            self.optimizer_acoustic, self.optimizer_visual,
            self.scheduler_mmilb, self.scheduler_main, self.scheduler_text, 
            self.scheduler_acoustic, self.scheduler_visual) = make_optimizers(self.model, self.text_enc, self.acoustic_enc, self.visual_enc, self.hp, 0)

        # Load checkpoint if it exists
        PATH = f"checkpoint_text{self.suffix}.pt"
        if os.path.exists(PATH):
            print("Loading from checkpoint")
            checkpoint = torch.load(PATH) 
            self.text_enc.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_text.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        PATH = f"checkpoint_acoustic{self.suffix}.pt"
        if os.path.exists(PATH):
            print("Loading from checkpoint")
            checkpoint = torch.load(PATH) 
            self.acoustic_enc.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_acoustic.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        PATH = f"checkpoint_visual{self.suffix}.pt"
        if os.path.exists(PATH):
            print("Loading from checkpoint")
            checkpoint = torch.load(PATH) 
            self.visual_enc.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_visual.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        PATH = f"checkpoint_main{self.suffix}.pt"
        if os.path.exists(PATH):
            print("Loading from checkpoint")
            checkpoint = torch.load(PATH) 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_main.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.scores = pickle.load(open(f'results/results{self.suffix}.pkl', 'rb'))


    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        # Main training loop

        model = self.model
        text_enc = self.text_enc
        acoustic_enc = self.acoustic_enc
        visual_enc = self.visual_enc
        optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main
        optimizer_visual = self.optimizer_visual
        optimizer_acoustic = self.optimizer_acoustic
        optimizer_text = self.optimizer_text

        scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main
        scheduler_text = self.scheduler_text
        scheduler_visual = self.scheduler_visual
        scheduler_acoustic = self.scheduler_acoustic

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, text_enc, acoustic_enc, visual_enc, optimizer, optimizer_visual, optimizer_acoustic, optimizer_text, criterion, epoch, stage, cpus, lrs, local_epochs_prev):
            epoch_loss = 0

            model.train()
            text_enc.train()
            acoustic_enc.train()
            visual_enc.train()

            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()

            left_batch = self.update_batch

            mem_pos_tv = []
            mem_neg_tv = []
            mem_pos_ta = []
            mem_neg_ta = []
            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data

                # for mosei we only use 50% dataset in stage 1
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                    if self.hp.dataset=="ur_funny":
                        y = y.squeeze()
                
                batch_size = y.size(0)

                if stage == 0:
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:
                    mem = {'tv':{'pos':mem_pos_tv, 'neg':mem_neg_tv},
                            'ta':{'pos':mem_pos_ta, 'neg':mem_neg_ta},
                            'va': {'pos':mem_pos_va, 'neg':mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}

                # Get embeddings
                with torch.no_grad():
                    enc_word = text_enc(text, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
                    text_embedding = enc_word[:,0,:] # (batch_size, emb_size)
                    audio_embedding = acoustic_enc(audio, alens)
                    visual_embedding = visual_enc(visual, vlens)

                loss_tmp = 0
                nce_tmp = 0
                ba_tmp = 0

                def optimize_(text_out, audio_out, visual_out, y, mem, optimizer, model, local_model, loss_tmp, nce_tmp, ba_tmp, stage):
                    # optimizer function for a mini-batch

                    lld, nce, preds, pn_dic, H = model(text_out, visual_out, audio_out, y, mem)


                    left_batch = self.update_batch
                    if stage == 1:
                        y_loss = criterion(preds, y)
                        
                        if len(mem_pos_tv) < mem_size:
                            mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                            mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                            mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                            mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                            if self.hp.add_va:
                                mem_pos_va.append(pn_dic['va']['pos'].detach())
                                mem_neg_va.append(pn_dic['va']['neg'].detach())
                       
                        else: # memory is full! replace the oldest with the newest data
                            oldest = i_batch % mem_size
                            mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                            mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                            mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                            mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()

                            if self.hp.add_va:
                                mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                                mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                        if self.hp.contrast:
                            loss = y_loss + self.alpha * nce - self.beta * lld
                        else:
                            loss = y_loss
                        if i_batch > mem_size:
                            loss -= self.beta * H
                        #loss.backward(retain_graph=True)
                        loss.backward()
                        
                    elif stage == 0:
                        # maximize likelihood equals minimize neg-likelihood
                        loss = -lld
                        #loss.backward(retain_graph=True)
                        loss.backward()
                    else:
                        raise ValueError('stage index can either be 0 or 1')
                    
                    left_batch -= 1
                    if left_batch == 0:
                        left_batch = self.update_batch
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.hp.clip)
                        optimizer.step()
                    loss_tmp += loss.item() 
                    nce_tmp += nce.item()
                    ba_tmp += (-H - lld)
                    return loss_tmp, nce_tmp, ba_tmp
        
                ## reset optimizers
                #(optimizer_main, optimizer_mmilb, optimizer_text, optimizer_acoustic, optimizer_visual,
                #    scheduler_mmilb, scheduler_main, scheduler_text, 
                #    scheduler_acoustic, scheduler_visual) = make_optimizers(model, text_enc, acoustic_enc, visual_enc, self.hp, epoch)

                # Local iterations with current version of server model
                local_epochs = np.abs((1-cpus[:,(int(epoch*num_batches+i_batch) % cpus.shape[1])])*19+1).astype(int)
                local_epochs_text = local_epochs[0]
                local_epochs_audio = local_epochs[1]
                local_epochs_visual = local_epochs[2]
                local_epochs_main = local_epochs[3]
                for g in optimizer_text.param_groups:
                    g['lr'] = lrs[0] 
                for g in optimizer_acoustic.param_groups:
                    g['lr'] = lrs[1] 
                for g in optimizer_visual.param_groups:
                    g['lr'] = lrs[2] 
                for g in optimizer.param_groups:
                    g['lr'] = lrs[3] 
                if optimizer_visual is not None:
                    for i in range(local_epochs_text):
                        text_enc.zero_grad()
                        # Text local iterations
                        enc_word = text_enc(text, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
                        text_out = enc_word[:,0,:] # (batch_size, emb_size)
                        loss_tmp, nce_tmp, ba_tmp = optimize_(text_out, audio_embedding, visual_embedding, y, mem, optimizer_text, model, text_enc, loss_tmp, nce_tmp, ba_tmp, stage)
                    for i in range(local_epochs_audio):
                        acoustic_enc.zero_grad()
                        # Audio local iterations
                        audio_out = acoustic_enc(audio, alens)
                        loss_tmp, nce_tmp, ba_tmp = optimize_(text_embedding, audio_out, visual_embedding, y, mem, optimizer_acoustic, model, acoustic_enc, loss_tmp, nce_tmp, ba_tmp, stage)
                    for i in range(local_epochs_visual):
                        visual_enc.zero_grad()
                        # Video local iterations
                        visual_out = visual_enc(visual, vlens)
                        loss_tmp, nce_tmp, ba_tmp = optimize_(text_embedding, audio_embedding, visual_out, y, mem, optimizer_visual, model, visual_enc, loss_tmp, nce_tmp, ba_tmp, stage)

                # Server local iterations with old embeddings
                for i in range(local_epochs_main):
                    model.zero_grad()
                    # Server local iterations
                    loss_tmp, nce_tmp, ba_tmp = optimize_(text_embedding, audio_embedding, visual_embedding, y, mem, optimizer, model, model, loss_tmp, nce_tmp, ba_tmp, stage)
                # Update learning rate
                if self.alg == "adapt":
                    for client in range(4):
                        lrs[client] *= local_epochs_prev[client]/local_epochs[client]
                        local_epochs_prev[client] = local_epochs[client]

                loss_tmp /= (local_epochs_main+local_epochs_text+local_epochs_audio+local_epochs_visual)
                nce_tmp /= (local_epochs_main+local_epochs_text+local_epochs_audio+local_epochs_visual)
                ba_tmp /= (local_epochs_main+local_epochs_text+local_epochs_audio+local_epochs_visual)
                
                proc_loss += loss_tmp * batch_size
                proc_size += batch_size
                epoch_loss += loss_tmp * batch_size
                nce_loss += nce_tmp * batch_size
                ba_loss += ba_tmp * batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size
                    avg_ba = ba_loss / proc_size
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, 'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                        avg_loss, avg_nce, avg_ba))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    start_time = time.time()
                    
            return epoch_loss / self.hp.n_train, lrs, local_epochs

        def evaluate(model, text_enc, acoustic_enc, visual_enc, criterion, test=False):
            # Get training and test accuracy

            model.eval()
            text_enc.eval()
            acoustic_enc.eval()
            visual_enc.eval()

            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset == 'iemocap':
                            y = y.long()
                    
                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    enc_word = text_enc(text, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
                    text = enc_word[:,0,:] # (batch_size, emb_size)
                    audio = acoustic_enc(audio, alens)
                    visual = visual_enc(vision, vlens)
                    _, _, preds, _, _ = model(text, visual, audio)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience
        scores = self.scores

        if self.start_epoch == 1: 
            test_loss, results, truths = evaluate(model, text_enc, acoustic_enc, visual_enc, criterion, test=True)
            if self.hp.dataset in ["mosei_senti", "mosei"]:
                curr_score = eval_mosei_senti(results, truths, True, 0, 0)
            elif self.hp.dataset == 'mosi':
                curr_score = eval_mosi(results, truths, True, 0, 0)
            elif self.hp.dataset == 'iemocap':
                curr_score = eval_iemocap(results, truths)
            scores.append(curr_score)

        # Load cpus
        cpus = np.load('../google-cpu-full.npy')
        rand_inds = np.random.randint(12476, size=(4)).tolist()
        cpus = cpus[rand_inds]
        cpus[cpus < 0] = 0
        cpus[cpus > 1] = 1
        lrs = []
        local_epochs = [] 
        for client in range(4):
            if client == 0:
                lrs.append(np.abs(self.hp.lr_bert/(19*np.max(1-cpus[client])+1)))
            else:
                lrs.append(np.abs(self.hp.lr_main/(19*np.max(1-cpus[client])+1)))
            local_epochs.append((19*np.max(1-cpus[client])+1).astype(int))
        print(local_epochs)
        print(lrs)

        for epoch in range(self.start_epoch, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # maximize likelihood
            #if self.hp.contrast:
            #    train_loss = train(model, text_enc, acoustic_enc, visual_enc, optimizer_mmilb, None, None, None, criterion, 0)

            # minimize all losses left
            train_loss, lrs, local_epochs = train(model, text_enc, acoustic_enc, visual_enc, optimizer_main, optimizer_visual, optimizer_acoustic, optimizer_text, criterion, epoch, 1, cpus, lrs, local_epochs)

            val_loss, _, _ = evaluate(model, text_enc, acoustic_enc, visual_enc, criterion, test=False)
            test_loss, results, truths = evaluate(model, text_enc, acoustic_enc, visual_enc, criterion, test=True)
            
            end = time.time()
            duration = end-start

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    
                    best_results = results
                    best_truths = truths
                    #print(f"Saved model at pre_trained_models/MM.pt!")
                    #save_model(self.hp, model)
            else:
                patience -= 1
            #    if patience == 0:
            #        break
                #if patience == 0: 
                #    self.hp.lr_mmilb *= 0.5
                #    self.hp.lr_main *= 0.5
                #    self.hp.lr_bert *= 0.5
                #    patience = self.hp.patience

            #(optimizer_main, optimizer_mmilb, optimizer_text, optimizer_acoustic, optimizer_visual,
            #    scheduler_mmilb, scheduler_main, scheduler_text, 
            #    scheduler_acoustic, scheduler_visual) = make_optimizers(model, text_enc, acoustic_enc, visual_enc, self.hp, epoch)
            #scheduler_main.step(val_loss)    # Decay learning rate by validation loss
            #scheduler_text.step(val_loss)    # Decay learning rate by validation loss
            #scheduler_visual.step(val_loss)    # Decay learning rate by validation loss
            #scheduler_acoustic.step(val_loss)    # Decay learning rate by validation loss

            if self.hp.dataset in ["mosei_senti", "mosei"]:
                curr_score = eval_mosei_senti(results, truths, True, train_loss, val_loss)
            elif self.hp.dataset == 'mosi':
                curr_score = eval_mosi(results, truths, True, train_loss, val_loss)
            elif self.hp.dataset == 'iemocap':
                curr_score = eval_iemocap(results, truths)
            scores.append(curr_score)
            pickle.dump(scores, open(f'results/results{self.suffix}.pkl', 'wb'))

            PATH = f"checkpoint_text{self.suffix}.pt"
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': text_enc.state_dict(),
                        'optimizer_state_dict': optimizer_text.state_dict(),
                        'loss': 0,
                        }, PATH)
            PATH = f"checkpoint_acoustic{self.suffix}.pt"
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': acoustic_enc.state_dict(),
                        'optimizer_state_dict': optimizer_acoustic.state_dict(),
                        'loss': 0,
                        }, PATH)
            PATH = f"checkpoint_visual{self.suffix}.pt"
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': visual_enc.state_dict(),
                        'optimizer_state_dict': optimizer_visual.state_dict(),
                        'loss': 0,
                        }, PATH)
            PATH = f"checkpoint_main{self.suffix}.pt"
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_main.state_dict(),
                        'loss': 0,
                        }, PATH)


        #print(f'Best epoch: {best_epoch}')
        #if self.hp.dataset in ["mosei_senti", "mosei"]:
        #    eval_mosei_senti(best_results, best_truths, True)
        #elif self.hp.dataset == 'mosi':
        #    self.best_dict = eval_mosi(best_results, best_truths, True)
        #elif self.hp.dataset == 'iemocap':
        #    eval_iemocap(results, truths)       
        sys.stdout.flush()
