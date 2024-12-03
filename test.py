from utils.predictor import predict

# TODO: Try for all models!!!!!
def use_model(model_name, vectorizer_name, title, perex):    
    print("For title: " + title + "\nand perex: " + perex + "\nthe sport is: ",end="")
    print(predict(model_name, vectorizer_name, title, perex))