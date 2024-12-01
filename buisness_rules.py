def get_model_by_buisness_rules(model_dict, context, is_multi_model_enabled):
    if is_multi_model_enabled:
        if context == "Главная":
            return model_dict["decision_tree"]
        elif context == "Важно":
            return model_dict["kmeans"]
        elif context == "Способ подписания":
            return model_dict["decision_tree"]
    return model_dict["catboost"]
