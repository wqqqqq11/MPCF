import torch
import os
import io


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    # 统一保存为best_model.pt，不再根据数据集名称区分
        name = 'best_model'
    
    # 使用指定的绝对路径
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pre_trained_models'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        model_path = os.path.join(save_dir, f'{name}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")


def load_model(args, name=''):
    # 统一加载best_model.pt，不再根据数据集名称区分
        name = 'best_model'
    
    # 使用指定的绝对路径
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pre_trained_models'))
        model_path = os.path.join(save_dir, f'{name}.pt')
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        model = torch.load(buffer)
        print(f"模型已从 {model_path} 加载")
        return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t
