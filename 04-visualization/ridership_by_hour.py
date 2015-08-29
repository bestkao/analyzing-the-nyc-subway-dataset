import pandasql
from ggplot import *

def plot_weather_data(turnstile_weather):
    '''
    You are passed in a dataframe called turnstile_weather.
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.
    You should feel free to implement something that we discussed in class
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/

    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    To see all the columns and data points included in the turnstile_weather
    dataframe.

    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''

    df = turnstile_weather[['Hour', 'ENTRIESn_hourly']]

    q = """
        SELECT Hour AS hour,
               sum(ENTRIESn_hourly)/count(*) AS hourlyentries
        FROM df
        GROUP BY hour
        """

    #Execute SQL command against the pandas frame
    rainy_days = pandasql.sqldf(q.lower(), locals())


    print ggplot(rainy_days, aes('hour', 'hourlyentries')) + \
            geom_bar(fill = '#cc2127', stat='bar') + \
            scale_x_continuous(name="Hour",
                               breaks=[0, 1, 2, 3, 4, 5,
                                       6, 7, 8, 9, 10, 11,
                                       12, 13, 14, 15, 16, 17,
                                       18, 19, 20, 21, 22, 23],
                               labels=['12AM', '1AM', '2AM', '3AM', '4AM', '5AM',
                                       '6AM', '7AM', '8AM', '9AM', '10AM', '11AM',
                                       '12PM', '1PM', '2PM', '3PM', '4PM', '5PM',
                                       '6PM', '7PM', '8PM', '9PM', '10PM', '11PM']) + \
            ggtitle("Average ENTRIESn_hourly by Hour") + \
            ylab("ENTRIESn_hourly")
