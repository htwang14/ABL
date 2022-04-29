from models.selector import *
from utils.util import *
from data_loader import *
from config import get_arguments
from backdoor_unlearning import test
    
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
    log_root = os.path.join(opt.log_root, '%s_%s_ABL_unlearning.txt' % (opt.dataset, opt.trigger_type))
    print("Epoch %d: Test_clean_acc %.4f | Test_bad_acc %.4f | Test_clean_loss %.4f | Test_bad_loss %.4f\n" % (
        epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))  

if __name__ == '__main__':
    opt = get_arguments().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, checkpoint_epoch = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=os.path.join('weight/unlearning_model', 'CIFAR10_badnet_grid_unlearning_epochs9.pth'),
                                n_classes=opt.num_class)
    model_ascent.to(opt.device)
    print('Finish loading ascent model...')

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    test_clean_loader, test_bad_loader = get_test_loader(opt)
    test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, 0)