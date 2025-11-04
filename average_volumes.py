import pandas as pd
import matplotlib.pyplot as plt

def main():
    file = "Amazon (AMZN) From 1997 To Dec-2024.csv"
    data = pd.read_csv(file)
    dates = data["Date"]
    dates_new = [2020, 2021,2022,2023,2024]
    volumes = [0,0,0,0,0]
    yearly_averages = [0,0,0,0,0]
    current_year = 2020
    for i in range(len(dates)):
        if int(dates[i][:4]) < current_year:
            continue
        if int(dates[i][:4]) == current_year:
            volumes[current_year-2020] += data["Volume"][i]
            yearly_averages[current_year-2020] += 1
        current_year += 1

    for i in range(len(volumes)):
        yearly_averages[i] = volumes[i] / yearly_averages[i]

    plt.plot(dates_new, yearly_averages)
    plt.show()
    plt.xlabel("Date")
    plt.ylabel("Volume")


if __name__ == "__main__":
    main()

