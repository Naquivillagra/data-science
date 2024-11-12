from sklearn.pipeline import Pipeline
from transformers import (
    RemoveDuplicatesTransformer,
    FilterDataTransformer,
    BoolDataTransformer,
    DateColsTransformer,
    TargetTransformer
)


def pre_hotel(reader, dumper):
    """Preprocess data pipeline."""
    pipeline = Pipeline(
        [
            # Transformers applied to raw data
            ("filter_duplicated_rows", RemoveDuplicatesTransformer()),
            ("filter_data", FilterDataTransformer()),
            ("convert_to_binary_columns", BoolDataTransformer()),
            ("convert_to_date", DateColsTransformer()),
            ("create_target", TargetTransformer()),

        ]
    )