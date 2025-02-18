from pathlib import Path
import os
import json
import csv
# FNN_SMALL_ACC = 0.9658
# FNN_BIG_ACC = 0.9718
# CNN_SMALL_ACC = 0.9827
RESNET152 = 0.6825
WRN = 0.6436

# construct the function to read log from every tool
def read_patchrepair_result():
    DIR_PATH = '/root/PatchART/results/tiny_imagenet/patch'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['WRN', 'resnet152']:
        result_dict[net] = {}
        for radius in [2, 4]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}


                for item in ['drawdown', 'rsr', 'rgr','time']:
                    result_dict[net][radius][data_number][item] = 0
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                if net == 'wide_resnet101_2':
                    net = 'WRN'
                radius = name[1]
                data_number = str(name[3])
                



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-2]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    # radius
                    # line = lines[24]
                    # line = line.split(' ')
                    # # remove the \n
                    # # get the radius
                    # radius = float(line[-1])
                    # radius = str(radius)
                    # get the drawdown
                    line = lines[-8]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    if net == 'WRN':
                        drawdown = WRN - drawdown
                    elif net == 'resnet152':
                        drawdown = RESNET152 - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-7]
                    line = line.split(' ')
                    rs = f'{float(line[-1])*100:.1f}'
                    # get the gene
                    line = lines[-6]
                    line = line.split(' ')
                    rgr = f'{float(line[-1])*100:.1f}'
                    # get the time
                    line = lines[-2]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rsr'] = rs
                    result_dict[net][radius][data_number]['rgr'] = rgr
                    result_dict[net][radius][data_number]['time'] = time

    # save the result to json file
    with open('patchrepair_tinyimagenet_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict   


def read_care_result():
    DIR_PATH = '/root/PatchART/tools/tinyimagenet/care'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['WRN', 'resnet152']:
        result_dict[net] = {}
        for radius in [2, 4]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rsr', 'rgr','time','dsr']:
                    if item != 'dsr':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        if net == 'WRN':
                            # full 0
                            result_dict[net][radius][data_number][item] = '0.0'
                        elif net == 'resnet152':
                            result_dict[net][radius][data_number][item] = '0.0'
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                # judge the net
                if net == 'wide_resnet_101_2':
                    net = 'WRN'
                elif net == 'resnet152':
                    net = 'resnet152'

                data_number = str(name[1])
                radius = name[2]
                # remove the .log
                radius = radius[:-4]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                    # resnet pgd result
                    # if net == 'resnet152':
                    #     line = lines[-1]
                    #     line = line.split(' ')
                    #     line = line[-2].split(':')
                    #     pgd = float(line[-1])
                    #     pgd = f'{pgd*100:.1f}'
                    #     lines = lines[:-1]
                    # # remove the white line
                    # lines = lines[:-2]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    
                    # get the drawdown
                    line = lines[2]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    
                    if net == 'WRN':
                        drawdown = WRN - drawdown
                    elif net == 'resnet152':
                        drawdown = RESNET152 - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[1]
                    line = line.split(' ')
                    rs = 1. - float(line[-3][:-1])
                    rs = f'{rs*100:.1f}'
                    # get the gene
                    line = lines[2]
                    line = line.split(' ')
                    gene = 1. - float(line[-3][:-1])
                    gene = f'{gene*100:.1f}'
                    # get the time
                    line = lines[3]
                    line = line.split(' ')
                    time1 = line[-1]
                    time1 = time1[:-2]
                    time1 = float(time1)

                    line = lines[4]
                    line = line.split(' ')
                    time2 = line[-1]
                    time2 = time2[:-2]
                    time2 = float(time2)
                    time = time1 + time2
                    time = f'{time:.1f}'
                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rsr'] = rs
                    result_dict[net][radius][data_number]['rgr'] = gene
                    result_dict[net][radius][data_number]['time'] = time
#                    result_dict[net][radius][data_number]['dsr'] = pgd
    # save the result to json file
    with open('care_tinyimagenet_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict     

def convert_dict_to_csv():
    '''
    convert the result to csv file
    
    '''
    import csv
    # read the result
    patchrepair_result = read_patchrepair_result()
    # art_result = read_art_result()
    care_result = read_care_result()
    # prdnn_result = read_prdnn_result()
    # reassure_result = read_reassure_result()
    # aprnn_result = read_aprnn_result()
    # construct the header
    # patchrepair_label_result = read_patchrepair_label_result()
    header = ['net', 'radius', 'data_number', 'tool', 'drawdown', 'rsr', 'rgr', 'time']
    # construct the csv file
    with open('tinyimagenet_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # write the patchart result
        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    for patch_format in ['PatchRepair']:
                        drawdown = patchrepair_result[net][radius][data_number]['drawdown']
                        rs = patchrepair_result[net][radius][data_number]['rs']
                        gene = patchrepair_result[net][radius][data_number]['gene']
                        time = patchrepair_result[net][radius][data_number]['time']
                        row = [net, radius, data_number, patch_format, drawdown, rs, gene, time]
                        writer.writerow(row)
        # write the art result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             drawdown = art_result[net][radius][data_number]['drawdown']
        #             rs = art_result[net][radius][data_number]['rs']
        #             gene = art_result[net][radius][data_number]['gene']
        #             time = art_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'art', drawdown, rs, gene, time]
        #             writer.writerow(row)




        # write the care result
        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    drawdown = care_result[net][radius][data_number]['drawdown']
                    rs = care_result[net][radius][data_number]['rs']
                    gene = care_result[net][radius][data_number]['gene']
                    time = care_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'care', drawdown, rs, gene, time]
                    writer.writerow(row)
        # write the prdnn result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             drawdown = prdnn_result[net][radius][data_number]['drawdown']
        #             rs = prdnn_result[net][radius][data_number]['rs']
        #             gene = prdnn_result[net][radius][data_number]['gene']
        #             time = prdnn_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'prdnn', drawdown, rs, gene, time]
        #             writer.writerow(row)
        # write the reassure result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             drawdown = reassure_result[net][radius][data_number]['drawdown']
        #             rs = reassure_result[net][radius][data_number]['rs']
        #             gene = reassure_result[net][radius][data_number]['gene']
        #             time = reassure_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'reassure', drawdown, rs, gene, time]
        #             writer.writerow(row)
        
     # write the aprnn result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             drawdown = aprnn_result[net][radius][data_number]['drawdown']
        #             rs = aprnn_result[net][radius][data_number]['rs']
        #             gene = aprnn_result[net][radius][data_number]['gene']
        #             time = aprnn_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'aprnn', drawdown, rs, gene, time]
        #             writer.writerow(row)
    # write the patchart label result

    # with open('tinyimagenet_label_result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     # write the patchart result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             # for patch_format in ['label']:
        #             drawdown = patchrepair_label_result[net][radius][data_number]['drawdown']
        #             rs = patchrepair_label_result[net][radius][data_number]['rs']
        #             gene = patchrepair_label_result[net][radius][data_number]['gene']
        #             time = patchrepair_label_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'label', drawdown, rs, gene, time]
        #             writer.writerow(row)
        # write the art result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             drawdown = art_result[net][radius][data_number]['drawdown']
        #             rs = art_result[net][radius][data_number]['rs']
        #             gene = art_result[net][radius][data_number]['gene']
        #             time = art_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'art', drawdown, rs, gene, time]
        #             writer.writerow(row)




        # write the care result
        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    drawdown = care_result[net][radius][data_number]['drawdown']
                    rs = care_result[net][radius][data_number]['rs']
                    gene = care_result[net][radius][data_number]['gene']
                    time = care_result[net][radius][data_number]['time']
                    row = [ net, radius, data_number, 'care', drawdown, rs, gene, time]
                    writer.writerow(row)


    # write another csv file for pgd result
    header = ['net', 'radius', 'data_number', 'tool', 'autoattack_success']
    pgd_r = pgd_result()

    with open('tinyimagenet_pgd_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    for tool in ['patch', 'ori', 'trades', 'label', 'care', 'prdnn', 'aprnn']:
                        pgd = pgd_r[net][radius][data_number][tool]['dsgr']
                        row = [net, radius, data_number, tool, pgd]
                        writer.writerow(row)
        # # write the aprnn result
        # for net in ['WRN', 'resnet152']:
        #     for radius in [2, 4]:
        #         radius = str(radius)
        #         for data_number in [500,1000]:
        #             data_number = str(data_number)
        #             pgd = aprnn_result[net][radius][data_number]['dsgr']
        #             row = [net, radius, data_number, 'aprnn', pgd]
        #             writer.writerow(row)

    # write another csv file for generalization result
    header = ['net', 'radius', 'data_number', 'tool', 'dsgr']
    generalization_r = generalizaiton_result()
    with open('tinyimagenet_generalization_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    for tool in ['PatchRepair', 'original', 'adv_training', 'label', 'care', 'prdnn', 'aprnn']:
                        defense = generalization_r[net][radius][data_number][tool]['dsgr']
                        row = [net, radius, data_number, tool, defense]
                        writer.writerow(row)


def read_trades_result():
    file_path = '/root/PatchART/tools/tinyimagenet/trade-tinyimagenet/result.txt'
    result_dict = {}
    for net in ['WRN', 'resnet152']:
        result_dict[net] = {}
        for radius in [2, 4]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                # for patch_format in ['small', 'big']:
                #     result_dict[net][radius][data_number][patch_format] = {}
                for item in ['drawdown', 'rsr', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
    
    
    
    
    with open(file_path, 'r') as f:
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
                item = a[0]
                value = a[1][:-1]
                value = float(value)
                if item == 'test':
                    if net == 'WRN':
                        value = WRN - value
                    elif net == 'resnet152':
                        value = RESNET152 - value
                    value = f'{value*100:.1f}'
                    item = 'drawdown'
                elif item == 'attack':
                    item = 'rgr'
                    value = f'{value*100:.1f}'
                elif item == 'repair':
                    item = 'rsr'
                    value = f'{value*100:.1f}'
                elif item == 'time':
                    item = 'time'
                    value = f'{value:.1f}'
                else:
                    continue
                result_dict[net][radius][data_number][item] = value

        
    # save the result to json file
    with open('trade_tinyimagenet10_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def pgd_result():
    
    pgd_result_file = '/root/PatchART/results/tiny_imagenet/repair/autoattack/compare_DSR.txt'
    
    care_result = read_care_result()
    # prdnn_result = read_prdnn_result()
    # aprnn_result = read_aprnn_result()
    
    
    result_dict = {}
    for net in ['WRN', 'resnet152']:
        result_dict[net] = {}
        for radius in [2, 4]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['patch', 'trades', 'care']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsr']:
                        if tool != 'care' and tool != 'prdnn' and tool != 'aprnn':
                            result_dict[net][radius][data_number][tool][item] = '--'
                        elif tool == 'care':
                            result_dict[net][radius][data_number][tool][item] = '0.0'
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
            if line[-1] == ' ':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'adv-train':
                    tool = 'trades'
                pgd = float(a[1][:-1])
                # pgd = 1- float(pgd)/float(data_number)
                pgd = f'{pgd*100:.1f}'
                if net ==  'wide_resnet101_2':
                    net = 'WRN'
                result_dict[net][radius][data_number][tool]['dsr'] = pgd

    # save the result to json file
    with open('pgd_tinyimagenet_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

def generalizaiton_result():
    pgd_result_file = '/root/PatchART/results/tiny_imagenet/repair/generalization/compare/compare_generalization.txt'
    result_dict = {}
    for net in ['WRN', 'resnet152']:
        result_dict[net] = {}
        for radius in [2, 4]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['PatchRepair', 'original', 'trades', 'care']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dgsr']:
                        if tool != 'care':
                            result_dict[net][radius][data_number][tool][item] = '--'
                        else:
                            # because the pgd of repair data is 0
                            result_dict[net][radius][data_number][tool][item] = '0.0'
    with open(pgd_result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[2][:-1]
            if net == 'wide_resnet101_2':
                net = 'WRN'
            radius = line[4][:-1]
            data_number = line[6][:-1]

            line = line[7:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'adv-train':
                    tool = 'trades'
                pgd = a[1][:-1]
                pgd = float(pgd)
                pgd = f'{pgd*100:.1f}'
                result_dict[net][radius][data_number][tool]['dgsr'] = pgd

    # save the result to json file
    with open('pgd_generlization_tinyimagenet10_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def convert_dict_to_csv_rq():
    patchrepair_result = read_patchrepair_result()
    care_result = read_care_result()
    trades_result = read_trades_result()
    pgd_r = pgd_result()
    generalization_r = generalizaiton_result()

    # with open('tinyimagenet_rq1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')    
    #     writer.writerow(['Model','radius', 'num', 'RSR/%', 'RGR/%', 'DD/%'])
    #     writer.writerow(['', '', '', 'CARE', 'PRDNN', 'APRNN', 'TRADES', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'TRADES', 'Ours', 'CARE', 'PRDNN', 'APRNN',  'TRADES', 'Ours'])
    #     for net in ['WRN', 'resnet152']:
    #         for radius in [2, 4]:
    #             radius = str(radius)
    #             for data_number in [500,1000]:
    #                 data_number = str(data_number)

    #                 rs_care = care_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_prdnn = prdnn_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_aprnn = aprnn_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_trades = trades_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_patchrepair = patchrepair_result[net][radius][data_number]['rs']+ '[]'

    #                 rgr_care = care_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_prdnn = prdnn_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_aprnn = aprnn_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_trades = trades_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_patchrepair = patchrepair_result[net][radius][data_number]['rgr'] + '[]'

    #                 dd_care = care_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_prdnn = prdnn_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_aprnn = aprnn_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_trades = trades_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_patchrepair = patchrepair_result[net][radius][data_number]['drawdown'] + '[]'                    


    #                 writer.writerow([net, radius, data_number, rs_care, rs_prdnn, rs_aprnn, rs_trades, rs_patchrepair, rgr_care, rgr_prdnn, rgr_aprnn, rgr_trades, rgr_patchrepair, dd_care, dd_prdnn, dd_aprnn, dd_trades, dd_patchrepair])


    with open('Table5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['', '', 'WRN-101-2', '', '', '', 'resnet152', '', '', '',])
        writer.writerow(['', '','2','', '4', '','2', '','4','',])
        writer.writerow(['', '', '500', '1000',  '500', '1000','500', '1000','500', '1000'])



        for item in ['rsr', 'rgr', 'drawdown']:
            for tool in [ 'care', 'trades','ours']:
                if item == 'rsr':
                    row = ['RSR/%']
                elif item == 'rgr':
                    row = ['RGR/%']
                elif item == 'drawdown':
                    row = ['DD/%']
                row.append(tool)
                for net in ['WRN', 'resnet152']:
                    for radius in [2, 4]:
                        radius = str(radius)
                        for data_number in [500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = patchrepair_result[net][radius][data_number][item] + '[]'
                            # elif tool == 'label':
                            #     value = patchrepair_label_result[net][radius][data_number][item]
                            elif tool == 'care':
                                value = care_result[net][radius][data_number][item] + '[]'
                            elif tool == 'trades':
                                value = trades_result[net][radius][data_number][item] + '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            row.append(value)
                writer.writerow(row)
        for item in [ 'dsr']:
            
            for tool in [ 'care', 'trades', 'ours']:
                row = ['DSR/%']
                row.append(tool)
                for net in ['WRN', 'resnet152']:
                    for radius in [2, 4]:
                        radius = str(radius)
                        for data_number in [500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = pgd_r[net][radius][data_number]['patch'][item]+ '[]'
                            elif tool == 'trades':
                                value = pgd_r[net][radius][data_number][tool][item]+ '[]'
                            else:
                                value = pgd_r[net][radius][data_number][tool][item]+ '[]'
                            # elif tool == 'label':
                            row.append(value)
                writer.writerow(row)
        for item in [ 'dgsr']:
            
            for tool in [ 'care','trades', 'ours']:
                row = ['DGSR/%']
                row.append(tool)
                for net in ['WRN', 'resnet152']:
                    for radius in [2, 4]:
                        radius = str(radius)
                        for data_number in [500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = generalization_r[net][radius][data_number]['PatchRepair'][item] + '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            # elif tool == 'label':
                            row.append(value)
                writer.writerow(row)
        for item in ['time']:
            
            for tool in [ 'care', 'trades', 'ours']:
                row = ['Time/s']
                row.append(tool)
                for net in ['WRN', 'resnet152']:
                    for radius in [2, 4]:
                        radius = str(radius)
                        for data_number in [500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = patchrepair_result[net][radius][data_number][item] + '[]'
                            # elif tool == 'label':
                            #     value = patchrepair_label_result[net][radius][data_number][item]
                            elif tool == 'care':
                                value = care_result[net][radius][data_number][item]+ '[]'
                            elif tool == 'trades':
                                value = trades_result[net][radius][data_number][item]+ '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            row.append(value)
                writer.writerow(row) 






def convert_dict_to_rq_transpose():
    patchrepair_result = read_patchrepair_result()
    care_result = read_care_result()
    trades_result = read_trades_result()
    pgd_r = pgd_result()
    generalization_r = generalizaiton_result()

    with open('Table5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow(['Model', 'r', 'n', 'RSR/%', 'RSR/%', 'RSR/%', 'RGR/%','RGR/%','RGR/%', 'DD/%','DD/%','DD/%', 'DSR/%','DSR/%','DSR/%', 'DGSR/%','DGSR/%','DGSR/%', 'Time/s','Time/s','Time/s'])
        writer.writerow(['', '','','CARE', 'TRADE', 'Ours', 'CARE', 'TRADE', 'Ours', 'CARE', 'TRADE', 'Ours','CARE', 'TRADE', 'Ours','CARE', 'TRADE', 'Ours','CARE', 'TRADE', 'Ours'])
        for net in ['WRN', 'resnet152']:
            for radius in [2, 4]:
                radius = str(radius)
                for data_number in [500,1000]:
                    data_number = str(data_number)
                    row = [net, radius, data_number]
                    for item in ['rsr', 'rgr', 'drawdown', 'dsr', 'dgsr', 'time']:
                        if item == 'rsr' or item == 'rgr' or item == 'drawdown' or item == 'time':
                            for tool in ['care', 'trades', 'ours']:
                                if tool == 'ours':
                                    value = patchrepair_result[net][radius][data_number][item]
                                    row.append(value)
                                elif tool == 'trades':
                                    value = trades_result[net][radius][data_number][item]
                                    row.append(value)
                                elif tool == 'care':
                                    value = care_result[net][radius][data_number][item]
                                    row.append(value)
                        elif item == 'dsr':
                            for tool in ['care', 'trades', 'ours']:
                                if tool == 'ours':
                                    value = pgd_r[net][radius][data_number]['patch'][item]
                                    row.append(value)
                                elif tool == 'trades':
                                    value = pgd_r[net][radius][data_number][tool][item]
                                    row.append(value)
                                else:
                                    value = pgd_r[net][radius][data_number][tool][item]
                                    row.append(value)   
                        elif item == 'dgsr':                                 
                            for tool in ['care', 'trades', 'ours']:
                                if tool == 'ours':
                                    value = generalization_r[net][radius][data_number]['PatchRepair'][item]
                                    row.append(value)
                                elif tool == 'trades':
                                    value = generalization_r[net][radius][data_number][tool][item]
                                    row.append(value)
                                else:
                                    value = generalization_r[net][radius][data_number][tool][item]
                                    row.append(value)
                    writer.writerow(row)
    

                        

if __name__ == '__main__':
    # patchrepair_result = read_patchrepair_result()
    # prdnn_result = read_prdnn_result()
    # care_result = read_care_result()
    # aprnn_result = read_aprnn_result()
    # pgd_result()
    # generalizaiton_result()

    # patch_repair_label_result = read_patchrepair_label_result()
    # convert_dict_to_csv()
    convert_dict_to_rq_transpose()
