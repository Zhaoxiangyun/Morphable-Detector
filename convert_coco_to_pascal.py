import argparse
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a COCO dataset into Pascal VOC format"
    )
    parser.add_argument(
        "path_coco_json", type=str, help="Path to the COCO json annotation file"
    )
    parser.add_argument("path_coco_images", type=str, help="Path to the COCO images")
    parser.add_argument(
        "path_voc_output", type=str, help="Path to store the annotations in VOC format"
    )
    return parser.parse_args()


def read_coco_data(path_json, path_images):
    assert path_json.endswith(".json")
    with open(path_json) as fid:
        data_json = json.load(fid)

    setname = os.path.basename(path_json)[:-5]
    if setname.startswith("instances_"):
        setname = setname[10:]

    catid2name = {cat["id"]: cat["name"] for cat in data_json["categories"]}

    # Convert the data into a per-image structure
    data_perimage = []
    for ii, img in enumerate(data_json["images"]):
        abspath = os.path.join(path_images, img["file_name"])
        symname = "__".join(
            [setname]
            + ["{:08d}".format(img["id"])]
            + [img["file_name"].replace("/", "__")]
        )
        anno = [tok for tok in data_json["annotations"] if tok["image_id"] == img["id"]]
        data_perimage.append(
            {
                "abspath": abspath,
                "symname": symname,
                "anno": anno,
                "imgheight": img["height"],
                "imgwidth": img["width"],
            }
        )
#        if ii % 50 == 0:
##            print(
##                "\rReading coco data: {:d}/{:d}".format(
##                    ii + 1, len(data_json["images"])
##                ),
##                end="",
##                flush=True,
##            )
#    print("")

    data = {
        "json": data_json,
        "setname": setname,
        "per_image": data_perimage,
        "catid2name": catid2name,
    }
    return data


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_voc_directory_structure(basepath):
    mkdir(os.path.join(basepath, "ImageSets", "Main"))
    mkdir(os.path.join(basepath, "JPEGImages"))
    mkdir(os.path.join(basepath, "Annotations"))


def write_voc_image_header(fid, name, img_filename, height, width):
    fid.write("  <folder>{:s}</folder>\n".format(name))
    fid.write("  <filename>{:s}</filename>\n".format(img_filename))
    fid.write("  <source>\n")
    fid.write("    <database>{:s}</database>\n".format(name))
    fid.write("    <annotation>{:s}</annotation>\n".format(name))
    fid.write("    <image>{:s}</image>\n".format(name))
    fid.write("    <flickrid>unknown</flickrid>\n")
    fid.write("  </source>\n")
    fid.write("  <owner>\n")
    fid.write("    <url>unknown</url>\n")
    fid.write("  </owner>\n")
    fid.write("  <size>\n")
    fid.write("    <width>{:d}</width>\n".format(width))
    fid.write("    <height>{:d}</height>\n".format(height))
    fid.write("    <depth>3</depth>\n")
    fid.write("  </size>\n")
    fid.write("  <segmented>0</segmented>\n")


def write_voc_objects(fid, catid2name, anno):
    for obj in anno:
        category_name = catid2name[obj["category_id"]]
        xmin, ymin, objw, objh = obj["bbox"]
        xmax = xmin + objw - 1
        ymax = ymin + objh - 1
        fid.write("  <object>\n")
        fid.write("    <name>{:s}</name>\n".format(category_name))
        fid.write("    <pose>unknown</pose>\n")
        fid.write("    <truncated>0</truncated>\n")
        fid.write("    <difficult>0</difficult>\n")
        fid.write("    <bndbox>\n")
        fid.write("      <xmin>{:d}</xmin>\n".format(xmin))
        fid.write("      <ymin>{:d}</ymin>\n".format(ymin))
        fid.write("      <xmax>{:d}</xmax>\n".format(xmax))
        fid.write("      <ymax>{:d}</ymax>\n".format(ymax))
        fid.write("    </bndbox>\n")
        fid.write("  </object>\n")


def create_voc_annotation_file(path_output, setname, catid2name, img_data):
    img_filename = img_data["symname"]
    anno_filename = img_filename[:-4] + ".xml"
    anno_filename = os.path.join(path_output, "Annotations", anno_filename)

    with open(anno_filename, "w") as fid:
        fid.write('<?xml version="1.0" ?>\n<annotation>')
        write_voc_image_header(
            fid, setname, img_filename, img_data["imgheight"], img_data["imgwidth"]
        )
        write_voc_objects(fid, catid2name, img_data["anno"])
        fid.write("</annotation>")


def write_coco_data_in_voc(coco_data, path_output):
    image_set_file = os.path.join(
        path_output, "ImageSets", "Main", coco_data["setname"] + ".txt"
    )
    fid = open(image_set_file, "w")
    for ii, img in enumerate(coco_data["per_image"]):
        # Add image to image_set_file
        fid.write("{:s}\n".format(img["symname"]))
        # Create a symlink for the image
        sympath = os.path.join(path_output, "JPEGImages", img["symname"])
        if not os.path.exists(sympath):
            os.symlink(img["abspath"], sympath)
        # Create annotation file
        create_voc_annotation_file(
            path_output, coco_data["setname"], coco_data["catid2name"], img
        )
#        if ii % 50 == 0:
#            print(
#                "\rWriting voc data: {:d}/{:d}".format(
#                    ii + 1, len(coco_data["per_image"])
#                ),
#                end="",
#                flush=True,
#            )
#    print("")
    fid.close()


def main():
#    args = parse_arguments()

#    create_voc_directory_structure(args.path_voc_output)
#
#    coco_data = read_coco_data(args.path_coco_json, args.path_coco_images)
#
#    write_coco_data_in_voc(coco_data, args.path_voc_output)
    create_voc_directory_structure("/net/acadia9a/data/xzhao/datasets/val17")

    coco_data = read_coco_data("/net/acadia1a/data/samuel/coco17/annotations/instances_val2017.json", "/net/acadia1a/data/samuel/coco17/val2017")

    write_coco_data_in_voc(coco_data, "/net/acadia9a/data/xzhao/datasets/val17")



if __name__ == "__main__":
    main()
