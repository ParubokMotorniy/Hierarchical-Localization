from .utils.parsers import parse_image_lists
from pathlib import Path
import pycolmap
import tqdm
from . import logger
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval

# queries_path = Path("/home/yarish_pn/POSELIB/Hierarchical-Localization/datasets/aachen/queries/*_time_queries_with_intrinsics.txt")
# queries = parse_image_lists(queries_path, with_intrinsics=True)
# for query in queries:
#     qname = '/'.join(query[0].split('/')[1:])
#     print(qname)
reference_sfm = Path("/home/yarish_pn/POSELIB/Hierarchical-Localization/outputs/aachen/sfm_superpoint+superglue")
queries = Path("/home/yarish_pn/POSELIB/Hierarchical-Localization/outputs/aachen/list_query.txt")
retrieval = Path("/home/yarish_pn/POSELIB/Hierarchical-Localization/outputs/aachen/pairs-query-netvlad50.txt")

queries = parse_image_lists(queries, with_intrinsics=False)
retrieval_dict = parse_retrieval(retrieval)

if not isinstance(reference_sfm, pycolmap.Reconstruction):
    reference_sfm = pycolmap.Reconstruction(reference_sfm)
db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

for qname, qcam in tqdm(queries):
    if qname not in retrieval_dict:
        logger.warning(f"No images retrieved for query image {qname}. Skipping...")
        continue
    db_names = retrieval_dict[qname]
    db_ids = []
    for n in db_names:
        if n not in db_name_to_id:
            logger.warning(f"Image {n} was retrieved but not in database")
            continue
        db_ids.append(db_name_to_id[n])

    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        print(f"\n Image {i}'s data: {[image.id, *image.qvec, *image.tvec, image.camera_id, image.name]} \n")

    