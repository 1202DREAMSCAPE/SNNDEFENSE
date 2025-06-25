import h5py

enhanced_model_path = "enhanced_weights/enhanced_CEDAR.h5"
with h5py.File(enhanced_model_path, 'r') as f:
    print("Keys in the model file:", list(f.keys()))
    if 'training_config' in f:
        training_config = f['training_config']
        print("Training configuration:", training_config.attrs)