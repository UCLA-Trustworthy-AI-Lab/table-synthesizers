from sklearn.preprocessing import label_binarize, MinMaxScaler,OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split


_UNSEEN_CLASS_LABEL = {"string":'unseen_classes', "number":-114514}

class DataPreprocessor:
    def __init__(self, target_column=None):
        self.encoders = {}
        self.scalers = {}
        self.target_column = target_column
        self.target_label_type = None

    def fit_transform_train(self, df, metadata=None):
        df_preprocessed = df.copy()
        # print("metadata is:",metadata)
        # print("df_preprocessed.columns:",df_preprocessed.columns)

        for column_name, dtype in metadata.items():
            # print("Preprocessing training column ", column_name, " with datatype ", dtype)
            try:
                if column_name not in df_preprocessed.columns:
                    continue

                # Only encode column name when it is categorical
                elif column_name == self.target_column and dtype in ["categorical", "binary"]:
                    # print("fitting target!")
                    le = LabelEncoder()
                    # Enable an unseen column
                    target_value_for_fitting = list(df_preprocessed[column_name].values.ravel())
                    # Avoid mixing string/number
                    # if df_preprocessed[column_name].dtype == 'object':
                    #    target_value_for_fitting += [_UNSEEN_CLASS_LABEL['string']]
                    #    self.target_label_type = "string"
                    # else:
                    #    target_value_for_fitting += [_UNSEEN_CLASS_LABEL['number']]
                    #    self.target_label_type = "number"
                    le.fit(target_value_for_fitting)
                    df_preprocessed[column_name] = le.transform(df_preprocessed[column_name])
                    self.encoders[column_name] = le

                elif dtype in ["categorical", "binary"]:
                    # OneHotEncoder for categorical and binary data
                    oe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    transformed = oe.fit_transform(df[[column_name]])
                    self.encoders[column_name] = oe

                    # Drop the original column and add the new one-hot encoded columns
                    # print(f"Fitted oe for column {column_name}")
                    temp_df = pd.DataFrame(transformed,
                                           columns=[f"{column_name}_{category}" for category in oe.categories_[0]],
                                           index=df_preprocessed.index)
                    # print(f"Fitted temp_df for column {column_name}")
                    df_preprocessed = pd.concat([df_preprocessed.drop(column_name, axis=1), temp_df], axis=1)
                    # df_preprocessed = df_preprocessed.drop(column_name, axis=1)

                elif dtype in ["continuous", "ordinal", "bounded_numerical"]:
                    # Min-Max Standardization
                    scaler = MinMaxScaler()
                    df_preprocessed[column_name] = scaler.fit_transform(df_preprocessed[[column_name]])
                    self.scalers[column_name] = scaler

                elif dtype == "datetime":
                    # Convert to datetime and then apply Min-Max Standardization
                    df_preprocessed[column_name] = pd.to_datetime(df_preprocessed[column_name], errors='coerce')
                    scaler = MinMaxScaler()
                    df_preprocessed[column_name] = scaler.fit_transform(df_preprocessed[[column_name]])
                    self.scalers[column_name] = scaler

                elif dtype in ["name", "index"]:
                    # Drop the column
                    df_preprocessed.drop(column_name, axis=1, inplace=True)

            except Exception as e:
                print(f"Problem {e} in transforming train data for {column_name}!! dtype given is: {dtype}.")

        return df_preprocessed

    def transform_real_and_synth(self, real_df, synth_df, metadata=None):

        '''
        Preprocess the real and synthetic dataframes together, so that the same transformations are applied to both.

        Return:

            real_df_processed: Preprocessed real DataFrame
            synth_df_processed: Preprocessed synthetic DataFrame
        '''
        # Step 1: Concatenate the two DataFrames
        concatenated_df = pd.concat([real_df, synth_df], ignore_index=True)

        # Step 2: Preprocess the entire concatenated DataFrame
        processed_df = self.fit_transform_train(concatenated_df, metadata)

        # Step 3: Split the processed DataFrame back into two
        real_df_processed = processed_df.iloc[:len(real_df)]
        synth_df_processed = processed_df.iloc[len(real_df):]

        return real_df_processed, synth_df_processed


    def transform_test(self, df, metadata):
        df_preprocessed = df.copy()
        # print("df_preprocessed.columns:",df_preprocessed.columns)

        for column_name, dtype in metadata.items():
            try:
                if column_name not in df_preprocessed.columns:
                    continue
                elif dtype in ["categorical", "binary"]:
                    encoder = self.encoders.get(column_name)

                    if encoder:
                        # Not target
                        if isinstance(encoder, OneHotEncoder):
                            transformed = encoder.transform(df[[column_name]])

                            # Drop the original column and add the new one-hot encoded columns
                            # df_preprocessed = df_preprocessed.drop(column_name, axis=1)
                            temp_df = pd.DataFrame(transformed, columns=[f"{column_name}_{category}" for category in
                                                                         encoder.categories_[0]],
                                                   index=df_preprocessed.index)
                            df_preprocessed = pd.concat([df_preprocessed.drop(column_name, axis=1), temp_df], axis=1)
                        # target column
                        else:
                            # Ensure all classes are either seen or is _UNSEEN_CLASS_LABEL
                            # known_classes = set(encoder.classes_)
                            # print("known_classes:",known_classes)
                            # label_replacement = _UNSEEN_CLASS_LABEL[self.target_label_type]
                            # df_preprocessed[column_name] = df_preprocessed[column_name].#apply(
                            #    lambda x: x if x in known_classes else label_replacement
                            # )
                            df_preprocessed[column_name] = encoder.transform(
                                df_preprocessed[column_name].values.ravel())

                elif dtype in ["continuous", "ordinal", "bounded_numerical", "datetime"]:
                    scaler = self.scalers.get(column_name)
                    if scaler:
                        if dtype in ['datetime']:
                            df_preprocessed[column_name] = pd.to_datetime(df_preprocessed[column_name], errors='coerce')
                        df_preprocessed[column_name] = scaler.transform(df_preprocessed[[column_name]])

                # print("Transformed:",column_name)
            except Exception as e:
                print(f"Problem {e} in {column_name}!")

        return df_preprocessed
