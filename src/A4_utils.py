import math

import numpy as np
import cv2

"""BGR values for different colors"""
col_bgr = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


class CVText:
    def __init__(self, color='white', bkg_color='black', location=0, font=3,
                 size=0.8, thickness=1, line_type=2, offset=(5, 25)):
        self.color = color
        self.bkg_color = bkg_color
        self.location = location
        self.font = font
        self.size = size
        self.thickness = thickness
        self.line_type = line_type
        self.offset = offset

        self.help = {
            'font': 'Available fonts: '
                    '0: cv2.FONT_HERSHEY_SIMPLEX, '
                    '1: cv2.FONT_HERSHEY_PLAIN, '
                    '2: cv2.FONT_HERSHEY_DUPLEX, '
                    '3: cv2.FONT_HERSHEY_COMPLEX, '
                    '4: cv2.FONT_HERSHEY_TRIPLEX, '
                    '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                    '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                    '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; ',
            'location': '0: top left, 1: top right, 2: bottom right, 3: bottom left; ',
            'bkg_color': 'should be empty for no background',
        }


class CVConstants:
    interp_types = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_AREA,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    fonts = {
        0: cv2.FONT_HERSHEY_SIMPLEX,
        1: cv2.FONT_HERSHEY_PLAIN,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_COMPLEX,
        4: cv2.FONT_HERSHEY_TRIPLEX,
        5: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    line_types = {
        0: cv2.LINE_4,
        1: cv2.LINE_8,
        2: cv2.LINE_AA,
    }


def stack_images(img_list, grid_size=None, stack_order=0, borderless=1,
                 preserve_order=0, return_idx=0,
                 only_height=0, placement_type=0):
    n_images = len(img_list)

    if grid_size is None or not grid_size:
        n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
    else:
        n_rows, n_cols = grid_size

        if n_rows < 0:
            n_rows = int(np.ceil(n_images / n_cols))
        elif n_cols < 0:
            n_cols = int(np.ceil(n_images / n_rows))

    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    if only_height:
        width = 0

    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    # curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                stack_idx.append(_curr_img_id)
                if not borderless:
                    curr_img = resize_ar(curr_img, width, height)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    if curr_row.shape[0] < curr_img.shape[0]:
                        curr_row = resize_ar(curr_row, 0, curr_img.shape[0])
                    elif curr_img.shape[0] < curr_row.shape[0]:
                        curr_img = resize_ar(curr_img, 0, curr_row.shape[0])
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                if curr_row.shape[1] < stacked_img.shape[1]:
                    curr_row = resize_ar(curr_row, stacked_img.shape[1], 0, placement_type=placement_type)
                elif curr_row.shape[1] > stacked_img.shape[1]:
                    stacked_img = resize_ar(stacked_img, curr_row.shape[1], 0)

                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img


def vis_seg(seg_img, class_cols=None, resize_factor=None):
    if seg_img.dtype == np.dtype(bool):
        seg_img = seg_img.astype(np.uint8)
        if class_cols is None:
            class_cols = {
                0: 'black',
                1: 'white',
            }

    assert class_cols is not None, "class_cols must be provided for non-binary mask"

    vis_seg_img = np.zeros_like(seg_img)

    vis_seg_img = np.stack((vis_seg_img,) * 3, axis=2)

    for cls, col in class_cols.items():
        vis_seg_img[seg_img == cls] = col_bgr[col]

    if resize_factor is not None:
        vis_seg_img = resize_ar(vis_seg_img, resize_factor=resize_factor)

    return vis_seg_img


def annotate(img_list, text=None,
             fmt=None,
             grid_size=None,
             max_width=0, max_height=0,
             img_labels=None,
             width=0, height=0,
             border_size=2):
    """

    :param np.ndarray | list | tuple img_list:
    :param str text:
    :param CVText fmt:
    :param tuple(int) grid_size:
    :return:
    """

    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list, ]

    if width > 0 or height > 0:
        for k, img in enumerate(img_list):
            img_list[k] = resize_ar(img, width=width, height=height)

    if img_labels is not None:
        assert len(img_labels) == len(img_list), "img_labels and img_list must have same length"

    if fmt is None:
        """use default format"""
        fmt = CVText()

    size = fmt.size

    color = col_bgr[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    out_img_list = []

    for _id, _img in enumerate(img_list):
        if len(_img.shape) == 2:
            _img = np.stack([_img, ] * 3, axis=2)

        _img_h, _img_w = _img.shape[:2]

        if img_labels is not None:
            img_label = img_labels[_id]
            (text_width, text_height) = cv2.getTextSize(
                img_label, font,
                fontScale=fmt.size,
                thickness=fmt.thickness)[0]

            text_height += fmt.offset[1]
            text_width += fmt.offset[0]
            label_img = np.zeros((text_height, text_width), dtype=np.uint8)
            cv2.putText(label_img, img_label, tuple(fmt.offset),
                        font, size, color, fmt.thickness, line_type)

            if len(_img.shape) == 3:
                label_img = np.stack([label_img, ] * 3, axis=2)

            if text_width < _img_w:
                label_img = resize_ar(label_img, width=_img_w, height=text_height,
                                      only_border=2, placement_type=1)

            if border_size:
                # horizontal borders
                border_img = np.full((border_size, _img_w, 3), 255, dtype=np.uint8)
                img_list_label = [label_img, border_img, _img, border_img]
            else:
                img_list_label = [label_img, _img]
            _img = stack_images(img_list_label, grid_size=(-1, 1), preserve_order=1)

        if border_size:
            # vertical border
            border_img = np.full((_img.shape[0], border_size, 3), 255, dtype=np.uint8)
            _img = stack_images([_img, border_img], grid_size=(1, -1), preserve_order=1)

        out_img_list.append(_img)

    img_stacked = stack_images(out_img_list, grid_size=grid_size, preserve_order=1)

    if text is not None:
        if '\n' in text:
            text_list = text.split('\n')
        else:
            text_list = [text, ]

        max_text_width = 0
        text_height = 0
        text_heights = []

        for _text in text_list:
            (_text_width, _text_height) = cv2.getTextSize(_text, font, fontScale=fmt.size, thickness=fmt.thickness)[0]
            if _text_width > max_text_width:
                max_text_width = _text_width
            text_height += _text_height + 5
            text_heights.append(_text_height)

        text_width = max_text_width + 10
        text_height += 30

        text_img = np.zeros((text_height, text_width, 3), dtype=np.uint8)
        location = list(fmt.offset)

        for _id, _text in enumerate(text_list):
            cv2.putText(text_img, _text, tuple(location), font, size, color, fmt.thickness, line_type)
            location[1] += text_heights[_id] + 5

        if text_width < img_stacked.shape[1]:
            text_img = resize_ar(text_img, width=img_stacked.shape[1], height=text_height,
                                 only_border=2, placement_type=1)

        if border_size:
            border_img = np.full((border_size, img_stacked.shape[1], 3), 255, dtype=np.uint8)
            img_list_txt = [text_img, border_img, img_stacked]
        else:
            img_list_txt = [text_img, img_stacked]
        img_stacked = stack_images(img_list_txt, grid_size=(-1, 1), preserve_order=1,
                                   )
    if img_stacked.shape[0] > max_height > 0:
        img_stacked = resize_ar(img_stacked, height=max_height)

    if img_stacked.shape[1] > max_width > 0:
        img_stacked = resize_ar(img_stacked, width=max_width)

    return img_stacked


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    # if style == 'dotted':
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)


def draw_dotted_poly(img, pts, color, thickness=1):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def transform_bbox(bbox, resize_factor, start_row, start_col):
    ymin, xmin, ymax, xmax = bbox

    ymin = np.floor(ymin * resize_factor + start_row)
    ymax = np.ceil(ymax * resize_factor + start_row)

    xmin = np.floor(xmin * resize_factor + start_col)
    xmax = np.ceil(xmax * resize_factor + start_col)

    return np.asarray([ymin, xmin, ymax, xmax], dtype=bbox.dtype)


def draw_bbox(img, bbox, is_dotted, thickness, col, label, show_label):
    xmin, ymin, xmax, ymax = bbox

    if is_dotted:
        draw_dotted_rect(img, (xmin, ymin), (xmax, ymax), col, thickness=thickness)
    else:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), col, thickness=thickness)

    if show_label and label is not None:
        cv2.putText(img, f'{int(label):d}', (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col)


def denormalize_bbox(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    img_h, img_w = img.shape[:2]
    xmin, xmax = math.floor(xmin * (img_w - 1)), math.ceil(xmax * (img_w - 1))
    ymin, ymax = math.floor(ymin * (img_h - 1)), math.ceil(ymax * (img_h - 1))
    return (xmin, ymin, xmax, ymax)


def normalize_bbox(img, bbox):
    xmin, ymin, xmax, ymax = bbox

    img_h, img_w = img.shape[:2]
    xmin, xmax = xmin / (img_w - 1), xmax / (img_w - 1)
    ymin, ymax = ymin / (img_h - 1), ymax / (img_h - 1)

    return xmin, ymin, xmax, ymax


def vis_bboxes(img, bboxes, labels, cols, thickness, show_label, resize_factor, is_dotted=False):
    if resize_factor is not None:
        norm_bboxes = [normalize_bbox(img, bbox) for bbox in bboxes]
        img = resize_ar(img, resize_factor=resize_factor)
        bboxes = [denormalize_bbox(img, bbox) for bbox in norm_bboxes]

    for bbox, label, col in zip(bboxes, labels, cols):
        draw_bbox(img, bbox, is_dotted, thickness, col_bgr[col], label, show_label)

    return img


def get_size(width, height, src_height, src_width, src_aspect_ratio, only_shrink, only_border):
    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        if only_shrink and width > src_width:
            width = src_width
        if only_border == 1:
            height = src_height
        # elif only_border == 2:
        #     pass
        else:
            height = int(width / src_aspect_ratio)
    elif width <= 0:
        if only_shrink and height > src_height:
            height = src_height

        if only_border == 1:
            width = src_width
        # elif only_border == 2:
        #     pass
        else:
            width = int(height * src_aspect_ratio)

    return width, height


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=1, only_border=0,
              only_shrink=0, resize_factor=None,
              size=None, auto_max=1):
    if size is not None:
        width, height = size

    src_height, src_width = src_img.shape[:2]

    if resize_factor is not None:
        width, height = int(src_width * resize_factor), int(src_height * resize_factor)

    src_aspect_ratio = float(src_width) / float(src_height)

    if len(src_img.shape) == 3:
        n_channels = src_img.shape[2]
    else:
        n_channels = 1

    if only_border == 2:
        assert width > 0 and height > 0, \
            "both width and height must be provided for strict only_border mode"
        assert src_width <= width and src_height <= height, \
            "source size must be <= target size for strict only_border mode"
    else:
        if auto_max and width > 0 and height > 0:
            width1, height1 = get_size(width, 0, src_height, src_width, src_aspect_ratio, only_shrink, only_border)
            width2, height2 = get_size(0, height, src_height, src_width, src_aspect_ratio, only_shrink, only_border)

            height_diff = height1 - height
            width_diff = width2 - width

            if height_diff > width_diff:
                width, height = width2, height2
            else:
                width, height = width1, height1
        else:
            width, height = get_size(width, height,
                                     src_height, src_width,
                                     src_aspect_ratio,
                                     only_shrink, only_border)

    aspect_ratio = float(width) / float(height)

    if only_border:
        dst_width = width
        dst_height = height
        if placement_type == 0:
            start_row = start_col = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
            start_col = int(dst_width - src_width)
        else:
            raise AssertionError('Invalid placement_type: {}'.format(placement_type))
    else:
        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            if placement_type == 0:
                start_row = 0
            elif placement_type == 1:
                start_row = int((dst_height - src_height) / 2.0)
            elif placement_type == 2:
                start_row = int(dst_height - src_height)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            if placement_type == 0:
                start_col = 0
            elif placement_type == 1:
                start_col = int((dst_width - src_width) / 2.0)
            elif placement_type == 2:
                start_col = int(dst_width - src_width)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=src_img.dtype)
    dst_img = dst_img.squeeze()

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    if not only_border:
        dst_img = cv2.resize(dst_img, (width, height))

    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, (resize_factor, start_row, start_col)
    else:
        return dst_img
