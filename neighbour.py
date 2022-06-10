import math

def read_attribute(file_path):
    attribute_dict = {}
    with open(file_path,"r") as file:
        lines = file.readlines()
        for line in lines:
            item_idx,attr_1,attr_2 = line.strip().split("|")
            attr_1 = int(attr_1) if attr_1 != "None" else -1
            attr_2 = int(attr_2) if attr_2 != "None" else -1
            attribute_dict[int(item_idx)] = [attr_1,attr_2]
    file.close()
    return attribute_dict

def k_neighbour_with_history(attribute_dict,history,item,k=3):
    if item not in attribute_dict:
        return []
    attr1,attr2 = attribute_dict[item]
    flag_1 = attr1!=-1
    flag_2 = attr2!=-1
    diff_list = []
    for idx,score in history.items():
        if idx not in attribute_dict:
            continue
        attributes = attribute_dict[idx]
        tmp_flag_1 = attributes[0]!=-1
        tmp_flag_2 = attributes[1]!=-1
        if flag_1==tmp_flag_1 and flag_2 == tmp_flag_2:
            diff = 0
            if flag_1:
                diff+=abs(attributes[0]-attr1)
            if flag_2:
                diff+=abs(attributes[1]-attr2)
            if diff<5000:
                diff_list.append([idx,diff])
                if len(diff_list)==k:
                    return diff_list
    return diff_list

def k_neighbour(attribute_dict,item,k=3):
    if item not in attribute_dict:
        return []
    attr1,attr2 = attribute_dict[item]
    flag_1 = attr1!=-1
    flag_2 = attr2!=-1
    diff_list = []
    for idx,attributes in attribute_dict.items():
        tmp_flag_1 = attributes[0]!=-1
        tmp_flag_2 = attributes[1]!=-1
        if flag_1==tmp_flag_1 and flag_2 == tmp_flag_2:
            diff = 0
            if flag_1:
                diff+=abs(attributes[0]-attr1)
            if flag_2:
                diff+=abs(attributes[1]-attr2)
            if diff==0:
                diff_list.append([idx,diff])
                if len(diff_list)==k:
                    return diff_list
    return diff_list

if __name__ == "__main__":
    attribute_dict = read_attribute("data-202205/itemAttribute.txt")
    k_neighbour(attribute_dict,0)
