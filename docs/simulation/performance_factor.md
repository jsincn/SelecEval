# Performance Factor

To reflect realistic device heterogeneity, we introduce a randomized performance factor that
acts as a multiplier on the expected run time. By incorporating this factor, we capture the
random usage patterns and energy states that prevent the devices from fully committing their
computational capabilities to the training process. Applying this performance
factor also significantly increases the difficulty in client selection. Even when devices with
appropriate estimated execution times are selected, there is still a chance that they may not
finish.

The performance factor is modeled as a normal distribution with a mean of 1 and a standard deviation of 0.2. These values are configurable by the user.