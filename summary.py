
# re-implementation of torchsummary
def Summary(model):
    def get_model_params_and_size(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return total_params, trainable_params, non_trainable_params

    total_params, trainable_params, non_trainable_params = get_model_params_and_size(model)
    
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("----------------------------------------------------------------")