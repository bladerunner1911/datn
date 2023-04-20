""" In train.py """
... 
from utils.plots import plot_evolve, plot_labels, plot_lr_scheduler
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_version, check_yaml, colorstr, get_latest_run,
                           increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer, 
                           SGDR, SGDR2, SGDR4)
...

# Scheduler - line 178
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif opt.sgdr:
        lf = SGDR(base_lr=1, T_i=30, T_mult=1, eta_min=hyp['lrf'])
    elif opt.sgdr1:
        lf = SGDR2(base_lr=1, T_0=1, T_i=1, T_mult=2, eta_min=hyp['lrf'])
    elif opt.sgdr2:
        lf = SGDR2(base_lr=1, T_0=10, T_i=10, T_mult=2, eta_min=hyp['lrf'])
    elif opt.sgdrwd:
        lf = SGDR4(base_lr=1, T_0=10, T_i=10, T_mult=2, decay_rate=0.8, eta_min=hyp['lrf'])
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=save_dir)

    
""" In utils/general.py - line 614 """
def SGDR(base_lr=0.01, T_i=10, T_mult=1, eta_min=0.0001):
    return lambda x: eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * (x % T_i) / T_i)) / 2 # x la so epoch
    
def SGDR2(base_lr=1., T_0=1, T_i=10, T_mult=1, eta_min=0.01):
    # return lambda x: eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
    return (lambda x: (
            eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * (x % T_0) / T_i)) / 2
            if (x >= T_0 and T_mult == 1)
            else (eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * SGDR2_helper(x, T_0, T_mult)[0] / SGDR2_helper(x, T_0, T_mult)[1])) / 2
                if (x >= T_0 and T_mult != 1)
                  else (eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * x / T_0)) / 2)
                  )
    ))
  
def SGDR2_helper(x, T_0, T_mult):
    n = int(math.log((x / T_0 * (T_mult - 1) + 1), T_mult))
    T_cur = x - T_0 * (T_mult ** n - 1) / (T_mult - 1)
    T_i = T_0 * T_mult ** (n)
    return T_cur, T_i
    
def SGDR3(x, base_lr=1., T_0=1., T_i=10., T_mult=1., eta_min=0.0001):
    # return lambda x: eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
    return (
        eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * (x % T_0) / T_i)) / 2
        if (x >= T_0 and T_mult == 1)
        else (eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * SGDR2_helper(x, T_0, T_mult)[0] / SGDR2_helper(x, T_0, T_mult)[1])) / 2
              if (x >= T_0 and T_mult != 1)
              else (eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * x / T_0)) / 2)
              )
    )

def SGDR4(base_lr=1., T_0=1, T_i=10, T_mult=1, decay_rate=0.8, eta_min=0.0001):
    return (lambda x: (
        SGDR3(x, base_lr, T_0, T_i, T_mult, eta_min) if x < 10 
        else (
            SGDR3(x, base_lr, T_0, T_i, T_mult, eta_min) * decay_rate if 10 <= x < 30
            else (
                SGDR3(x, base_lr, T_0, T_i, T_mult, eta_min) * decay_rate ** 2 if 30 <= x < 70 
                else (
                    SGDR3(x, base_lr, T_0, T_i, T_mult, eta_min) * decay_rate ** 3 if 70 <= x < 150 
                    else (
                        SGDR3(x, base_lr, T_0, T_i, T_mult, eta_min) * decay_rate ** 4
                    )
                )
            )
        ) 
    )
    )
