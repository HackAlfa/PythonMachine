import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from kneed import KneeLocator

# Определяем числовые и категориальные колонки
categorical_columns = ["segment", "role", 'context']
numerical_columns = ["organizations", "claims", "common_mobile", 
                        "common_web", "special_mobile", 
                        "special_web"]

def preprocess(df):
    df[['common_mobile', 'common_web', 'special_mobile', 'special_web']] = pd.json_normalize(df['signatures']).set_index(df.index)
    if "currentMethod" in df.columns:
        df = df.drop("currentMethod", axis=1)
    df = df.dropna()

    transformed_data = pd.DataFrame()

    scaler = StandardScaler()
    numerical_data = df[numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaled_data = scaler.fit_transform(numerical_data)
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)
    transformed_data = pd.concat([transformed_data, scaled_df], axis=1)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    for col in categorical_columns:
        encoded_cols = encoder.fit_transform(df[[col]])
        encoded_col_names = encoder.get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names)
        transformed_data = pd.concat([transformed_data, encoded_df], axis=1)
        
    df = df.dropna()
    transformed_data = transformed_data.dropna()
    return df, transformed_data

def elbow_method(transformed_data, is_plot_needed=False)->int:
    inertia = []
    range_clusters = range(1, 11)

    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(transformed_data)
        inertia.append(kmeans.inertia_)
    
    knee_locator = KneeLocator(range_clusters, inertia, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    if is_plot_needed:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range_clusters, inertia, marker='o')
        plt.title('Метод локтя для определения оптимального числа кластеров')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Сумма квадратов расстояний до центроидов (Inertia)')
        plt.xticks(range_clusters)
        plt.grid()
        plt.show()
    return optimal_k

def join_predicted_clusters(df, transformed_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    print(transformed_features.shape)
    df['predicted_cluster'] = kmeans.fit_predict(transformed_features)  # Добавляем кластер в DataFrame
    return df, kmeans


def get_cluster_summary(df, n_clusters):
    cluster_summary = {}

    for cluster in range(n_clusters):
        cluster_data = df[df['predicted_cluster'] == cluster]
        summary = {}
        
        for col in numerical_columns:
            summary[col] = cluster_data[col].median()
            summary[col + "q75"] = cluster_data[col].quantile(0.75)
            summary[col + "q25"] = cluster_data[col].quantile(0.25)
        
        for col in categorical_columns:
            summary[col] = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else None
        
        cluster_summary[cluster] = summary

    summary_df = pd.DataFrame(cluster_summary).T
    return summary_df


def join_predicted_tree(df, transformed_features):
    """Обучаем дерево решений для классификации ответов и джоиним к датафрейму"""
    X = transformed_features
    y = df["predicted_cluster"]
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    df['predicted_cluster_tree'] = tree.predict(transformed_features)    
    return df, tree, X, y


def explain_decision(tree, sample, feature_names):
    """
    Объясняет предсказание решающего дерева для одного объекта.
    :param tree: обученная модель DecisionTreeClassifier.
    :param sample: объект для предсказания (1D-массив).
    :param feature_names: список имен признаков.
    :return: строка с объяснением предсказания.
    """
    node_indicator = tree.decision_path([sample])
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    leaf_id = tree.apply([sample])[0]
    
    explanation = []
    for node_id in range(node_indicator.shape[1]):
        if not node_indicator[0, node_id]:
            continue
        if feature[node_id] != -2:  # Узел не является листом
            feature_name = feature_names[feature[node_id]]
            threshold_value = threshold[node_id]
            if sample[feature[node_id]] <= threshold_value:
                explanation.append(f"{feature_name} <= {threshold_value:.2f}")
            else:
                explanation.append(f"{feature_name} > {threshold_value:.2f}")
    
    predicted_class = tree.classes_[tree.tree_.value[leaf_id].argmax()]
    explanation_text = "\n".join(explanation)
    return explanation_text, predicted_class, tree.tree_.value[leaf_id]


def get_interpretesions(conditions:list[str], user_info):
    english_to_russian_cat = {"segment": "Ваш сегмент бизнеса", "role":"Вы"}
    "segment", "role"
    english_to_russian_num = {
        "organizations": "Количество организаций",
        "common_mobile": "Количество подписаний стандартных документов в мобайле",
        "common_web": "Количество подписаний стандартных документов в web",
        "special_mobile": "Количество подписаний важных документов в мобайле",
        "special_web": "Количество подписаний важных документов в web",
    }
    interpretension = []

    for condition in conditions.split("\n"):
        col_name, operator_value = condition.split(' ', 1)
        operator, value = operator_value.split(' ', 1)
        value = value.strip()
        categorical_column_name = col_name[:col_name.rfind('_')] if col_name.rfind('_') != -1 else col_name
        if categorical_column_name in categorical_columns:    
            interpretension.append(f"{english_to_russian_cat[categorical_column_name]} - {user_info[categorical_column_name]}")
        elif col_name in numerical_columns:
            interpretension.append(f"{english_to_russian_num[col_name]} в среднем = {int(float(value))} . У вас - {int(user_info[col_name])}")
    
    if len(interpretension) == 0:
        return None
            
    result = "\n-".join(interpretension)
    return result

def pipeline_kmeans_learning(df, is_summary_data_needed=False, is_tree_interpretetion_needed=False, is_kmeans_saving_needed=True):
    df = df.dropna()
    df, transformed_df = preprocess(df)
    optimal_k = elbow_method(transformed_df)
    answer, model_kmeans = join_predicted_clusters(df, transformed_df, optimal_k)
    if is_kmeans_saving_needed:
        with open("kmeans_model.pkl", "wb") as f:
            pickle.dump(model_kmeans, f)
    if is_summary_data_needed:
        summ = get_cluster_summary(answer, optimal_k) #if you want to analyze clusters
    if is_tree_interpretetion_needed:
        updated_df, tree, features, predicts = join_predicted_tree(answer, transformed_df) # присоединяем дерево решений. ОНО не совпадАЕТ с результатами k-means в 37 процентах случаев. Поэтому так не делаем
