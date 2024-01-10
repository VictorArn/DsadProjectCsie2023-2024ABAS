import numpy as np
import pandas as pd
import utils as utl
import EFA.efa as efa
import PCA.pca as pca
import factor_analyzer as fa
import visual as vi
from sklearn.preprocessing import StandardScaler


# compute Barlett sphericity test
bartlett_sphericity = fa.calculate_bartlett_sphericity(x=X_df)
print(bartlett_sphericity, type(bartlett_sphericity))
if bartlett_sphericity[0] > bartlett_sphericity[1]:
    print('There is at least one common factor!')
else:
    print('There are no common factors!')
    exit(-1)

# compute Kaiser-Meyer-Olkin (KMO) indices
kmo = fa.calculate_kmo(x=X_df)
print(kmo, type(kmo))
if kmo[1] > 0.5:
    print('There is at least one common factor that can be extracted!')
else:
    print('There are no common factors!')
    exit(-1)
# create the correlogram of KMO indices
vector = kmo[0]
print(vector, type(vector))
matrix = vector[:, np.newaxis]
print(matrix, type(matrix))
matrix_df = pd.DataFrame(data=matrix, index=varName,
                         columns=['KMO indices'])
# save KMO indices into a CSV file
matrix_df.to_csv('./dataOUT/KMO.csv')

vi.corelograma(matrice=matrix_df, titlu='Correlogram of KMO indices')
# vi.afisare()

# loop for extracting the relevant factors
no_of_factors = 1
chi2TabMin = 1
noOfSemnificantFactors = 1
for k in range(1, X.shape[1]):
# for k in range(1, 4):
    modelFA = fa.FactorAnalyzer(n_factors=k)
    modelFA.fit(X_df)
    commonFactors = modelFA.loadings_  # gives us the common factors
    specificFactors = modelFA.get_uniquenesses()
    print(commonFactors)
    print(specificFactors)
    modelEFA = efa.EFA(X)
    chi2Calc, chi2Tab = modelEFA.calculTestBartlett(commonFactors, specificFactors)
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2TabMin > chi2Tab:
        chi2TabMin = chi2Tab
        noOfSemnificantFactors = k

print('No. of significant factors: ', noOfSemnificantFactors)

# Create a FA model with noOfSemnificantFactors
fitModelFA = fa.FactorAnalyzer(n_factors=noOfSemnificantFactors)
fitModelFA.fit(X_df)
factorLoadingsFA = fitModelFA.loadings_
# save the correlation between the initial variables and the common factors extracted
factors = ['F'+str(j+1) for j in range(noOfSemnificantFactors)]
factorLoadingsFA_df = pd.DataFrame(data=factorLoadingsFA,
                                index=varName, columns=factors)
factorLoadingsFA_df.to_csv('./dataOUT/factorLoadingsFA.csv')

# create the correlogram of factor loadings from FA
vi.corelograma(matrice=factorLoadingsFA_df, titlu='Correlogram of FA factor loadings')
# vi.afisare()

# extract from Factor Analyzer instance the eigenvalues associated to the factors
eigenValuesFA = fitModelFA.get_eigenvalues()
print(eigenValuesFA[0])

# create the graphic of explained variance from FA
vi.componentePrincipale(valoriProprii=eigenValuesFA[0],
                         titlu='Explained variance of the FA extracted factors')
# vi.afisare()

# compute the pricipal components for the initial X matrix
modelPCA = pca.PCA(X)
eigenValuesEFA = modelPCA.getValProp()
factorLoadingsEFA = modelPCA.getRxc()  # these are the factor loadings
components = ['C'+str(j+1) for j in range(X.shape[1])]
factorLoadingsEFA_df = pd.DataFrame(data=factorLoadingsEFA,
                                    index=varName, columns=components)
# save the factor loadings provided by the PCA
factorLoadingsEFA_df.to_csv('./dataOUT/factorLoadingsEFA.csv')

# create the correlogram of factor loadings from PCA
vi.corelograma(matrice=factorLoadingsEFA_df, titlu='Correlogram of PCA factor loadings')
# vi.afisare()


# create the graphic of explained variance from PCA
vi.componentePrincipale(valoriProprii=eigenValuesEFA,
                         titlu='Explained variance of the PCA components')
vi.afisare()