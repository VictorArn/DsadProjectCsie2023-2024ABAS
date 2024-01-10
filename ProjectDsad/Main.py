import pandas as pd
import Function as fun
import PCA.pca as pca
import graphics as g


table = pd.read_csv('./dataIN/Infant_Mortality.csv', index_col=0)
print(table)

# create a list of useful variables
# vars = table.columns[1:]
vars = table.columns.values[0:]
# vars = list(table.columns.values[1:])
print(vars, type(vars))

# create a list of observations
obs = table.index.values
print(obs, type(obs))

# no. of variables
m = vars.shape[0]
print(m)
# no. of observations
n = len(obs)
print(n)

# create the matrix X of observed variables
X = table[vars].values
print(X.shape, type(X))

# standardise the X matrix
Xstd = fun.standardize(X)
print(Xstd.shape)
# save the standardised matrix into CSV file
Xstd_df = pd.DataFrame(data=Xstd, index=obs,
                       columns=vars)
print(Xstd_df)
Xstd_df.to_csv('./dataOUT/StdMatrix.csv')

# instantiate a PCA object
modelPCA = pca.PCA(Xstd)
alpha = modelPCA.getEigenValues()

g.principalComponents(eigenvalues=alpha)
# g.show()

# extract the principal components
prinComp = modelPCA.getPrinComp()
# save principal components into a CSV file
components = ['C'+str(j+1) for j in range(prinComp.shape[1])]
prinComp_df = pd.DataFrame(data=prinComp, index=obs,
                           columns=components)
prinComp_df.to_csv('./dataOUT/PCA.csv')

# extract the factor loadings
factorLoadings = modelPCA.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=vars,
                                 columns=components)
print(factorLoadings_df)
# save the factor loadings into a CSV file
factorLoadings_df.to_csv('./dataOUT/FactorLoadings.csv')
g.correlogram(matrix=factorLoadings_df, title='Correlogram of factor loadings')
# g.show()

# extract teh scores
scores = modelPCA.getScores()
scores_df = pd.DataFrame(data=scores, index=obs, columns=components)
# save the scores
scores_df.to_csv('./dataOUT/Scores.csv')
g.correlogram(matrix=scores_df, title='Correlogram of scores')
#g.show()

# extract the quality of points representation
qualObs = modelPCA.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, index=obs, columns=components)
# save the quality of points representation
qualObs_df.to_csv('./dataOUT/QualityObs.csv')
g.correlogram(matrix=qualObs_df, title='Correlogram of quality of points representation')
#g.show()

# extract the observations contribution to the axes' variance
contribObs = modelPCA.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs, index=obs, columns=components)
# save the observation contribution to the axes variance
contribObs_df.to_csv('./dataOUT/ContributionObs.csv')
g.correlogram(matrix=contribObs_df, title="Correlogram of observations contribution to the axes' variance")
# g.show()

# extract the commonalities
common = modelPCA.getCommon()
common_df = pd.DataFrame(data=common, index=vars, columns=components)
# save the commnalities
common_df.to_csv('./dataOUT/Commonalities.csv')
g.correlogram(matrix=common_df, title='Correlogram of commonalities')
g.show()
