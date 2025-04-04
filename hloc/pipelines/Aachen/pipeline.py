import argparse
from pathlib import Path
from pprint import pformat
import pycolmap

from ... import (
    colmap_from_nvm,
    extract_features,
    localize_sfm_poselib_sample,
    localize_sfm_colmap_sample,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)

solver_map = {"p3p":pycolmap.absolute_pose_estimation}

def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images_upright/"

    outputs = args.outputs  # where everything will be saved
    sift_sfm = outputs / "sfm_sift"  # from which we extract the reference poses
    reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build
    sfm_pairs = (
            outputs / f"pairs-db-covis{args.num_covis}.txt"
    )  # top-k most covisible in SIFT model
    loc_pairs = (
            outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    )  # top-k retrieved by NetVLAD
    results = outputs / f"Aachen_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    # list the standard configurations available
    logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    features = extract_features.main(feature_conf, images, outputs)

    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        sift_sfm,
    )
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    #functor = None
    #xtra_args = {}
    #if args.solver != "p3p":
        #functor = localize_sfm_poselib_sample.main
        #xtra_args["solver"] = args.solver
    #else:
        #functor = localize_sfm_colmap_sample.main
        #xtra_args["solver"] = solver_map[args.solver]

    #functor(reference_sfm,
            #dataset / "queries/*_time_queries_with_intrinsics.txt",
            #loc_pairs,
            #features,
            #loc_matches,
            #results,
            #covisibility_clustering=True,
            #prepend_camera_name=True,
            #**xtra_args)

    localize_sfm_poselib_sample.main(reference_sfm,
            dataset / "queries/*_time_queries_with_intrinsics.txt",
            loc_pairs,
            features,
            loc_matches,
            results,
            covisibility_clustering=True,
            prepend_camera_name=True,
            solver=args.solver
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Number of image pairs for loc, default: %(default)s",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="p3p",
        help="Solver to use for localization, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
