# Analysis
import numpy as np

# Statistics
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binom
from scipy.stats import friedmanchisquare, wilcoxon, chi2, pearsonr
from sklearn.metrics import r2_score    

# Builtin
import pdb


class Classifierbinomial():

## "data" is the observation matrix, where rows refer to inputs and columns to outputs.
## The cells in "key_matrix" should inform if the index-respective input-ouput match would be a correct pairing (1) or an incorrect one (0).
## In symmetric cases where matrix size is NxN (number of inputs equals the number of outputs), 
# the default solution matrix is an identity matrix.

    def __init__(self, data, key_matrix=None):

        self.data=data
        self.key_matrix=key_matrix

        if(data.shape[0]==data.shape[1] and np.array_equal(key_matrix, None)):
            self.key_matrix=np.identity(self.data.shape[0])
        elif(data.shape[0]!=data.shape[1] and np.array_equal(key_matrix, None)): 
            raise Exception('Solution matrix has not been defined')
        elif(self.data.shape!=self.key_matrix.shape):
            raise Exception('Dimensionality of solution matrix does not match data.')

        self.total=0
        self.correct_class=0
        self.null_probability=1/self.data.shape[1]
        # These are the n, k and p for the binomial distribution, in respective order.
        # Since columns refer to possible outputs, the probability of pairing an input with the right output by chance is 1/n(columns)

        for index, values in np.ndenumerate(self.data):
            if(self.key_matrix[index[0],index[1]]==1): 
                self.correct_class+=values
            self.total+=values
        # Read matrices, give n and k their observed values in this trial.

# returns the binomial tail probability for observing a classification sample under null hypothesis
    def binomial_probability(self):

        accuracy_probability = round(1 - binom.cdf(self.correct_class, self.total, self.null_probability), 3)
        return accuracy_probability

# returns the most credible interval for true accuracy, given the sample
    def accuracy_confidence_interval(self, confidence=0.05):

        conf_interval=(proportion_confint(count=self.correct_class, nobs=self.total, alpha=confidence, method='beta'))
        accuracy = self.correct_class / self.total
        lower_bound = round(conf_interval[0], 3)
        upper_bound = round(conf_interval[1], 3)
        return accuracy, lower_bound, upper_bound


class Statistics():
    def friedman_test(self, data):

        # Unpack the data columns into separate variables
        # This is a list of lists, where each list is a column of the data matrix
        data_columns = [data[:, i] for i in range(data.shape[1])]
        friedman_statistic, p_value = friedmanchisquare(*data_columns)
        return friedman_statistic, p_value

    def wilcoxon_test(self, data_1, data_2):

        wilcoxon_statistic, p_value = wilcoxon(data_1, data_2)
        return wilcoxon_statistic, p_value

    def chi_squared_test(self, data_1, data_2):

        chi_squared_statistic, p_value = chi2(data_1, data_2)
        return chi_squared_statistic, p_value

    def pearson_correlation(self, data_1, data_2):

        pearson_correlation, p_value = pearsonr(data_1, data_2)
        return pearson_correlation, p_value

    def goodness_of_fit(self, data1, data2):
        '''
        Return the coefficient of determination, i.e. the R2 score of the regression
        between data1 and data2
        '''
        return r2_score(data1, data2)

        
if __name__ == '__main__':

    data=np.matrix([[5,5,4,1],[2,3,2,1],[4,0,3,2]])
    key=np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    classtest=Classifierbinomial(data, key)

    print(f"p-value is: {classtest.binomial_probability()}")
    alpha_confidence=0.05
    accuracy, lower_bound, upper_bound = classtest.accuracy_confidence_interval(alpha_confidence)
    print(f"Confidence interval for true accuracy percentage is: {lower_bound, upper_bound} with {100-(alpha_confidence*100)} % certainty ")


       
    


