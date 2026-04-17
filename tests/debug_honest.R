library(HonestDiD)
library(foreach)
registerDoSEQ()
data(BCdata_EventStudy)

print("Testing with registerDoSEQ()...")
results <- createSensitivityResults(betahat        = BCdata_EventStudy$betahat,
                                    sigma          = BCdata_EventStudy$sigma,
                                    numPrePeriods  = length(BCdata_EventStudy$prePeriodIndices),
                                    numPostPeriods = length(BCdata_EventStudy$postPeriodIndices),
                                    alpha          = 0.05)
print(head(results))
