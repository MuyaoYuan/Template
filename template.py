def set_template(args):
    if args.template == 'resnet50':
        args.model = 'resnet50'
        args.n_colors = 1
        args.n_classes = 10
        args.epochs = 50
        args.batch_size = 20
        args.lr = 1e-4
        args.device_ids = [1, 2]
        args.autosave = 1
        
    else:
        print('Please Enter Appropriate Template!!!')
    
