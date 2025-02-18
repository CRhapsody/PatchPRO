import json
import csv
import os
# FNN_SMALL_ACC = 0.9658
# FNN_BIG_ACC = 0.9718
# CNN_SMALL_ACC = 0.9827
RESNET18 = 0.8833
VGG19 = 0.9342

# construct the function to read log from every tool
def read_patchrepair_result(size):
    if size == 'small':
        DIR_PATH = '/root/PatchART/results/cifar10/repair'
    elif size == 'big':
        DIR_PATH = '/root/PatchART/results/cifar10/big'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}


                for item in ['time', 'verified']:
                    result_dict[net][radius][data_number][item] = 0
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                radius = name[1]
                data_number = str(name[3])
                



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-3]

                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # get the verified
                    if size == 'small':
                        line = lines[-16]
                    elif size == 'big':
                        line = lines[-14]
                    line = line.split(' ')
                    if line[-1] != 'certified.\n':
                        verified = 'no'
                    else:
                        verified = 'yes'

                    # record the result
                    result_dict[net][radius][data_number]['time'] = time
                    result_dict[net][radius][data_number]['verified'] = verified
    # save the result to json file
    return result_dict

def pgd_result(size):
    if size == 'small':
        pgd_result_file = '/root/PatchART/results/cifar10/repair/autoattack/compare_autoattack_ac.txt'    
    elif size == 'big':   
        pgd_result_file = '/root/PatchART/results/cifar10/repair/autoattack/comapre_autoattack_ac_big.txt'    
    
    
    
    
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['patch', 'ori', 'trades', 'label', 'care', 'prdnn', 'aprnn']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsr']:
                            result_dict[net][radius][data_number][tool][item] = '--'

    with open(pgd_result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[1]
            radius = line[3]
            data_number = line[2]

            line = line[5:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'adv-train':
                    tool = 'trades'
                elif tool == 'big':
                    tool = 'patch'
                pgd = a[1][:-1]
                if size == 'small':
                    pgd = 1- float(pgd)/float(data_number)
                else:
                    pgd = float(pgd)/float(data_number)
                pgd = f'{pgd*100:.1f}'
                result_dict[net][radius][data_number][tool]['dsr'] = pgd

    # save the result to json file
    # with open('pgd_cifar_result.json', 'w') as f:
    #     json.dump(result_dict, f, indent=4)
    return result_dict

def generalizaiton_result(size):
    if size == 'small':
        pgd_result_file = '/root/PatchART/results/cifar10/repair/generalization/compare/compare_generalization.txt'
    elif size == 'big':
        pgd_result_file = '/root/PatchART/results/cifar10/repair/generalization/compare_big/compare_generalization.txt'
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['PatchRepair', 'trades', 'care', 'prdnn','aprnn']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsgr']:
                            result_dict[net][radius][data_number][tool][item] = '--'
    with open(pgd_result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[2][:-1]
            radius = line[4][:-1]
            data_number = line[6][:-1]

            line = line[7:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'label_repair':
                    tool = 'label'
                elif tool == 'adv_training':
                    tool = 'trades'
                pgd = a[1][:-1]
                pgd = float(pgd)
                pgd = f'{pgd*100:.1f}'
                result_dict[net][radius][data_number][tool]['dsgr'] = pgd

    # save the result to json file
    return result_dict

def convert_dict_to_csv_rq():
    patchrepair_result_small = read_patchrepair_result(size='small')
    patchrepair_result_big = read_patchrepair_result(size='big')
    pgd_result_small = pgd_result(size='small')
    pgd_result_big = pgd_result(size='big')
    generalizaiton_result_small = generalizaiton_result(size='small')
    generalizaiton_result_big = generalizaiton_result(size='big')
    # construct the csv file
    with open('Table6.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['', '', 'vgg19', '', '', '', '','', '', '','', '', '','resnet18', '', '', '', '', '', '',])
        writer.writerow(['', 'radius', '4','4','4','4','4', '8', '8','8','8', '8','4', '4','4','4', '4','8', '8','8','8', '8'])
        writer.writerow(['', '', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000'])
        for item in ['dsr', 'dsgr', 'time', 'verified']:
            if item == 'dsr':
                row = ['DSR/%']
            elif item == 'dsgr':
                row = ['DSGR/%']
            elif item == 'time':
                row = ['Time/s']
            elif item == 'verified':
                row = ['Verified']
            for size in ['small', 'big']:
                if size == 'small':
                    row.append('PS')
                    for net in ['vgg19', 'resnet18']:
                        for radius in [4, 8]:
                            for data_number in [50,100,200,500,1000]:
                                if item == 'dsr':
                                    row.append(pgd_result_small[net][str(radius)][str(data_number)]['patch'][item])
                                elif item == 'time':
                                    row.append(patchrepair_result_small[net][str(radius)][str(data_number)][item])
                                elif item == 'verified':
                                    row.append(patchrepair_result_small[net][str(radius)][str(data_number)][item])
                                else:
                                    row.append(generalizaiton_result_small[net][str(radius)][str(data_number)]['PatchRepair'][item])
                    writer.writerow(row)
                    row = row[:1]
                
                else:
                    row.append('PL')
                    for net in ['vgg19', 'resnet18']:
                        for radius in [4, 8]:
                            for data_number in [50,100,200,500,1000]:
                                if item == 'dsr':
                                    row.append(pgd_result_big[net][str(radius)][str(data_number)]['patch'][item])
                                elif item == 'time':
                                    row.append(patchrepair_result_big[net][str(radius)][str(data_number)][item])
                                elif item == 'verified':
                                    row.append(patchrepair_result_big[net][str(radius)][str(data_number)][item])
                                else:
                                    row.append(generalizaiton_result_big[net][str(radius)][str(data_number)]['PatchRepair'][item])
                    writer.writerow(row)
                        
            # save the result to csv file
            # with open('PL_and_PS_result.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter=',')
            #     writer.writerow(row)
    return
if __name__ == '__main__':
    convert_dict_to_csv_rq()
