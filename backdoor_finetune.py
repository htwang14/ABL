from models.selector import *
from utils.util import *
from data_loader import *
from config import get_arguments


def train_step_finetuing(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target, _) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target, _) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    # save training progress
    log_root = os.path.join(opt.log_root, '%s_%s_ABL_finetune.txt' % (opt.dataset, opt.trigger_type))
    fp = open(log_root, 'a+')
    fp.write("Epoch %d: Test_clean_acc %.4f | Test_bad_acc %.4f | Test_clean_loss %.4f | Test_bad_loss %.4f\n" % (
        epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    fp.flush()
    fp.close()


    return acc_clean, acc_bd


def train(opt):
    # mkdir:
    if not os.path.isdir(opt.finetune_model_root):
        os.makedirs(opt.finetune_model_root)

    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, checkpoint_epoch = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=os.path.join(opt.isolation_model_root, '%s_%s_tuning_epochs%d.pth' % (opt.dataset, opt.trigger_type, 15)),
                                n_classes=opt.num_class)
    model_ascent.to(opt.device)
    print('Finish loading ascent model...')

    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')
    data_path_isolation = os.path.join(opt.isolate_data_root, "%s_%s_isolation%s_examples.npy" % (opt.dataset, opt.trigger_type,
                                                                                                    opt.isolation_ratio))
    data_path_other = os.path.join(opt.isolate_data_root, "%s_%s_other%s_examples.npy" % (opt.dataset, opt.trigger_type,
                                                                                            1 - opt.isolation_ratio))

    tf_compose_finetuning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    tf_compose_unlearning = transforms.Compose([
        transforms.ToTensor()
    ])

    isolate_poisoned_data = np.load(data_path_isolation, allow_pickle=True)
    poisoned_data_tf = Dataset_npy(full_dataset=isolate_poisoned_data, transform=tf_compose_unlearning)
    isolate_poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      )

    isolate_other_data = np.load(data_path_other, allow_pickle=True)
    isolate_other_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_finetuning)
    isolate_other_data_loader = DataLoader(dataset=isolate_other_data_tf,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              )

    test_clean_loader, test_bad_loader = get_test_loader(opt)

    if opt.finetuning_ascent_model == True:
        # this is to improve the clean accuracy of isolation model, you can skip this step
        print('----------- Finetuning isolation model --------------')
        for epoch in range(0, opt.finetuning_epochs):
            learning_rate_finetuning(optimizer, epoch, opt)
            train_step_finetuing(opt, isolate_other_data_loader, model_ascent, optimizer, criterion,
                             epoch + 1)
            acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

            # save checkpoint at interval epoch
            if (epoch + 1) == opt.finetuning_epochs:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_ascent.state_dict(),
                    'clean_acc': acc_clean[0],
                    'bad_acc': acc_bad[0],
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(opt.finetune_model_root, '%s_%s_finetune_epochs%d.pth' % (opt.dataset, opt.trigger_type, epoch)))



def learning_rate_finetuning(optimizer, epoch, opt):
    if epoch < 40:
        lr = opt.lr_finetuning_init
    elif epoch < 60:
        lr = 0.01
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    train(opt)

if (__name__ == '__main__'):
    main()