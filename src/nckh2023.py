def has_helmet(x0_m, x1_m, y0_m, yc_m, xc_h, yc_h):
    """
        Kiem tra nguoi lai xe doi mu bao hiem
    """
    return x0_m < xc_h and xc_h < x1_m and y0_m < yc_h and yc_h < yc_m

def has_license_plate(x0_m, x1_m, yc_m, y1_m, xc_p, yc_p):
    """
        Kiem tra phuong tien co the nhan dien bien so
    """
    return x0_m < xc_p and xc_p < x1_m and yc_m < yc_p and yc_p < y1_m

def astype_int(array):
    temp = []
    for i in array:
        temp.append(int(i))
    return temp

def in_detection_zone(target_xy, zone_xyxy):
    """
        input: [int, int], [int, int, int, int]
        output: bool
    """
    return target_xy[0] > zone_xyxy[0] and target_xy[1] > zone_xyxy[1] and target_xy[0] < zone_xyxy[2] and target_xy[1] < zone_xyxy[3]
