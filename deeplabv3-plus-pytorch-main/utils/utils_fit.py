import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from nets.deeplabv3_training import CE_Loss, Dice_loss
from utils.utils import get_lr
from utils.utils_metrics import f_score, cal_mIou
# from utils.utils_metrics import cal_mIoU

DEVICE = 'cuda:0'


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, num_classes):
    total_loss      = 0
    total_f_score   = 0
    total_miou = 0
    total_oa = 0
    total_acc = 0
    # val_iou = []
    # val_mIoU_epochs = []

    val_loss        = 0
    val_f_score     = 0
    val_miou = 0
    val_oa = 0
    val_acc= 0

    model_train.train()
    model_train.to(DEVICE)
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            imgs, labels = batch
            imgs.cuda()
            labels.cuda()

            # with torch.no_grad():
            #     imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
            #     pngs    = torch.from_numpy(pngs).long()
            #     labels  = torch.from_numpy(labels).type(torch.FloatTensor)
            #     if cuda:
            #         imgs    = imgs.cuda()
            #         pngs    = pngs.cuda()
            #         labels  = labels.cuda()

            optimizer.zero_grad()
            outputs = model_train(imgs)
            # loss = nn.CrossEntropyLoss()
            # loss = loss(outputs, labels)
            loss = CE_Loss(outputs, labels, num_classes=num_classes).cuda()

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                #  -------------------------------#
                #   计算f_score
                #  -------------------------------#
                _f_score, miou, acc, oa = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            total_miou += miou
            total_acc += acc
            total_oa += oa
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'miou': total_miou/(iteration + 1),
                                'acc': total_acc / (iteration + 1),
                                'OA': total_oa/(iteration + 1)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    model_train.cuda()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, labels = batch

            with torch.no_grad():
                # imgs.cuda()
                # labels.cuda()
            #     imgs    = torch.from_numpy(imgs).type(torch.FloatTensor^)
            #     pngs    = torch.from_numpy(pngs).long()
            #     labels  = torch.from_numpy(labels).type(torch.FloatTensor)
            #     if cuda:
            #         imgs    = imgs.cuda()
            #         pngs    = pngs.cuda()
            #         labels  = labels.cuda()

                outputs = model_train(imgs)

                loss = CE_Loss(outputs, labels, num_classes = num_classes)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice
                #  -------------------------------#
                #   计算f_score
                #  -------------------------------#
                _f_score, miou,acc, oa = f_score(outputs, labels)

                val_loss += loss.item()
                val_f_score += _f_score.item()
                val_miou += miou
                val_acc += acc
                val_oa += oa
                # iou = cal_mIou(outputs, labels)
                # val_iou.append(iou)

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer),
                                'mIou': val_miou/(iteration+1),
                                'val_acc': val_acc/(iteration+1),
                                "val_OA": val_oa/(iteration+1)})
            pbar.update(1)
        # val_mIoU_epochs.append(np.mean((val_iou), dtype=float))
    loss_history.append_loss(total_loss/(epoch_step+1), val_loss/(epoch_step_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f || mIou: %.3f' % (total_loss / (epoch_step ), val_loss / (epoch_step_val), val_miou / (epoch_step_val )))
    print('savemodel:logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / (epoch_step ), val_loss / (epoch_step_val )))
