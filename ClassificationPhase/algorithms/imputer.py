from fancyimpute import KNN, IterativeImputer
import numpy as np

class KNNWrapper:
    # TODO: Must be improved to avoid unused parameters... Maybe use tuples?
    def __init__(self, images_test, images_with_mv_test, masks_test, data_split):
        self.imputer = KNN(5)
        
    def predict(self, images_with_mv_test, masks_test):
        #Impute the data in each image
        images = []
        for k in range(images_with_mv_test.shape[0]):
            if(k%100 == 0):
                print("KNN: Imputing Image " + str(k) + " of " + str(images_with_mv_test.shape[0]))
            test_image = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.imputer.fit_transform(test_image)
            images.append(imputed_image)
            
        imputed_data = np.expand_dims(np.asarray(images), axis=3)
            
        return imputed_data
        
class MICEWrapper:
    
    # TODO: Must be improved to avoid unused parameters... Maybe use tuples?
    def __init__(self, images_test, images_with_mv_test, masks_test, data_split):
        
        self.mice_impute = IterativeImputer()
        
    def predict(self, images_with_mv_test, masks_test):
        #Impute the data in each image
        images = []
        for k in range(images_with_mv_test.shape[0]):
            if(k%20 == 0):
                print("MICE: Imputing Image " + str(k) + " of " + str(images_with_mv_test.shape[0]))
            test_image = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.mice_impute.fit_transform(test_image)
            images.append(imputed_image)
            
        imputed_data = np.expand_dims(np.asarray(images), axis=3)
            
        return imputed_data