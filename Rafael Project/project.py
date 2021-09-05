import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
global fd

def prepare():
    global fd
    fd.pivot_table(index=fd.loc[0])
    fd.pop('targetName')
    routes_per_class = fd[['class']].value_counts()
    histograms = []
    for i in range(1, 17):
        i_rows = fd.loc[fd['class'] == i]
        distances = []
        for i in range(len(i_rows)):
            current_row = fd.iloc[[i]].values.tolist()[0]
            f_x = 2
            f_z = 4
            i = 44
            temp = current_row[44]
            while str(temp) != "nan" and i < 203:
                i += 7
                temp = current_row[i]
            l_x = i
            l_z = i + 2
            distances.append(math.dist([current_row[f_x], current_row[f_z]], [current_row[l_x], current_row[l_z]]))
        create_histogram(distances, len(i_rows), i)

def create_histogram(distance, rockets, rocket_class):
    num_bins = 100
    plt.hist(distance, num_bins)
    plt.show()

    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    # plt.plot(bins, y, '--', color ='black')

def draw_route(rockets):
    colors = ['b', 'g', 'r', 'c ', 'm', 'y', 'k', 'purple', 'orange', 'pink', 'black', 'brown', 'gray', 'lime', 'navy',
              'aqua']
    fig, ax = plt.subplots()
    for index, row in rockets.iterrows():
        x = []
        z = []
        i = 0
        while (~(pd.isnull(row[f'Time_{i}']))) and (i < 29):
            x.append(row[f'posX_{i}'])
            z.append(row[f'posZ_{i}'])
            i += 1
        ax.plot(x, z, '-', color=colors[int(row['class']) - 1])

def draw_energy(rockets):
    colors = ['b', 'g', 'r', '#1f77b4', 'm', 'y', 'k', 'purple', 'orange', 'pink', 'black', 'brown', 'gray', 'lime',
              'navy', 'aqua']
    fig, ax = plt.subplots()
    for index, row in rockets.iterrows():
        ax.plot(row.values.tolist(), '-', color=colors[int(row['class'] - 1)])

def draw_first_row():
    draw_route(fd.head(1))

def draw_50_routs():
    selected_rows = fd[fd['class'] < 7]
    draw_route(selected_rows.head(50))

def draw_50_routs_1_and_6():
    selected_rows = fd[((fd['class'] == 1) | (fd['class'] == 6)) & (fd['Time_29'] != 'NaN')]
    draw_route(selected_rows.head(50))

def draw_50_up_and_down():
    selected_rows = fd[((fd['class'] == 1) | (fd['class'] == 6)) & (fd['Time_29'].notnull()) & (fd['velZ_0'] > 0) & (
            fd['velZ_29'] < 0)]
    draw_route(selected_rows.head(50))

def prepare_data(rocket_type1, rocket_type2):
    train_set = fd[(fd['class'] == rocket_type1) | (fd['class'] == rocket_type2)]
    test_set = train_set.sample(int(0.2 * len(train_set)), replace=False)
    train_set = train_set[~train_set.index.isin(test_set.index)]
    class_col = test_set[['class']]
    test_set = test_set.drop(columns=['class'])
    train_set.to_csv(f'train_set_{rocket_type1}_{rocket_type2}.csv')
    test_set.to_csv(f'test_set_{rocket_type1}_{rocket_type2}.csv')
    class_col.to_csv(f'test_classes_{rocket_type1}_{rocket_type2}.csv')

def prepare_4_data(rocket_type1, rocket_type2, rocket_type3=None, rocket_type4=None):
    train_set = fd[(fd['class'] == rocket_type1) | (fd['class'] == rocket_type2) | (fd['class'] == rocket_type3) | (
            fd['class'] == rocket_type4)]
    test_set = train_set.sample(int(0.2 * len(train_set)), replace=False)
    train_set = train_set[~train_set.index.isin(test_set.index)]
    class_col = test_set.loc[:, ['class']]
    test_set = test_set.drop(columns=['class'])
    train_set.to_csv(f'train_set_{rocket_type1}_{rocket_type2}_{rocket_type3}_{rocket_type4}.csv')
    test_set.to_csv(f'test_set_{rocket_type1}_{rocket_type2}_{rocket_type3}_{rocket_type4}.csv')
    class_col.to_csv(f'test_classes_{rocket_type1}_{rocket_type2}_{rocket_type3}_{rocket_type4}.csv')

def last_time(row):
    for i in range(30):
        if (pd.isnull(row[f'Time_{i}'])):
            return i - 1
    return 29

def kinetic(row, i):
    return row[f'velX_{i}'] ** 2 + row[f'velY_{i}'] ** 2 + row[f'velZ_{i}'] ** 2

def potential(row, i):
    return 10 * row[f'posZ_{i}']

def determine_class(rocket_type1, rocket_type2):
    train_set = pd.read_csv(f'train_set_{rocket_type1}_{rocket_type2}.csv')
    columns = [f'sec_{i}' for i in range(30)] + ['class']
    kinetic_energy = pd.DataFrame(columns=columns)
    for index, row in train_set.iterrows():
        to_append = [kinetic(row, i) + potential(row, i) for i in range(30)] + [row['class']]
        kinetic_energy.loc[len(kinetic_energy)] = to_append
    draw_route(train_set)
    draw_energy(kinetic_energy)
    if rocket_type1 == 1 and rocket_type2 == 16:
        return detrmine(rocket_type1, rocket_type2, 0, 180000)

# def draw_vel(rocket_type1, rocket_type2):
#     train_set = pd.read_csv(f'train_set_{rocket_type1}_{rocket_type2}.csv')
#     columns = [f'velX_{i}' for i in range(30)] + ['class']
#     train_set_vel = train_set[columns]
#     draw_energy(train_set_vel)

def detrmine(rocket_type1, rocket_type2, range_min, range_max):
    rockets_class = []
    rockets = pd.read_csv(f'test_set_{rocket_type1}_{rocket_type2}.csv')
    for index, rocket in rockets.iterrows():
        energy_scala = [kinetic(rocket, i) + potential(rocket, i) for i in range(30)]
        if range_min <= max(energy_scala) <= range_max:
            rockets_class.append(rocket_type1)
        else:
            rockets_class.append(rocket_type2)
    class_col_1_16 = [item[-1] for item in pd.read_csv('test_classes_1_16.csv').values.tolist()]
    confusion_matrix_ = confusion_matrix(class_col_1_16, rockets_class)
    F1_score = f1_score(class_col_1_16, rockets_class)
    return F1_score

def determine_4_class(rocket_type1, rocket_type2, rocket_type3, rocket_type4):
    train_set = pd.read_csv(f'train_set_{rocket_type1}_{rocket_type2}_{rocket_type3}_{rocket_type4}.csv')
    # draw_route(train_set)
    columns = [f'sec_{i}' for i in range(30)] + ['class']
    energy = pd.DataFrame(columns=columns)
    for index, row in train_set.iterrows():
        to_append = [(10 * row[f'posZ_{i}'] + row[f'posZ_{i + 1}']) +
                     (2 * (pow(math.dist([row[f'posX_{i + 1}'], row[f'posY_{i + 1}'], row[f'posZ_{i + 1}']],
                                         [row[f'posX_{i}'], row[f'posY_{i}'], row[f'posZ_{i}']]), 2)))
                     for i in range(29)] + [row['class']]
        energy.loc[len(energy)] = to_append
    draw_energy(energy)

def ml(rocket_type1, rocket_type2, rocket_type3=None, rocket_type4=None):
    to_open = f'{rocket_type1}_{rocket_type2}'
    if rocket_type3 is not None:
        to_open += f'_{rocket_type3}_{rocket_type4}'
    print(f'types: {to_open}')
    train_set = pd.read_csv(f'train_set_{to_open}.csv')
    test_set = pd.read_csv(f'test_set_{to_open}.csv')
    test_classes = pd.read_csv(f'test_classes_{to_open}.csv').loc[:, ['class']]
    train_set.fillna(value=1, inplace=True)
    test_set.fillna(value=1, inplace=True)
    clf1 = LogisticRegression(max_iter=100000).fit(train_set.iloc[:, :-1], train_set['class'])
    score1 = clf1.score(test_set, test_classes)
    print('LogisticRegression: ', score1)
    clf2 = RandomForestClassifier(max_depth=100, random_state=0, n_estimators=128).fit(train_set.iloc[:, :-1], train_set['class'])
    score2 = clf2.score(test_set, test_classes)
    print('RandomForestClassifier: ', score2)

def ml1(rocket_type1, rocket_type2, rocket_type3=None, rocket_type4=None):
    to_open = f'{rocket_type1}_{rocket_type2}'
    if rocket_type3 is not None:
        to_open += f'_{rocket_type3}_{rocket_type4}'
    print(f'types: {to_open}')
    train_set = pd.read_csv(f'train_set_{to_open}.csv')
    test_set = pd.read_csv(f'test_set_{to_open}.csv')
    test_classes = pd.read_csv(f'test_classes_{to_open}.csv').loc[:, ['class']]
    train_set.fillna(value=1, inplace=True)
    test_set.fillna(value=1, inplace=True)
    columns = [f'sec_{i}' for i in range(30)]
    train_energy = pd.DataFrame(columns=columns)
    for index, row in train_set.iterrows():
        to_append = [kinetic(row, i) + potential(row, i) for i in range(30)]
        train_energy.loc[len(train_energy)] = to_append
    test_energy = pd.DataFrame(columns=columns)
    for index, row in test_set.iterrows():
        to_append = [kinetic(row, i) + potential(row, i) for i in range(30)]
        test_energy.loc[len(test_energy)] = to_append
    extended_train = pd.concat([train_energy, train_set], axis=1, join='inner')
    extended_test = pd.concat([test_energy, test_set], axis=1, join='inner')
    clf1 = LogisticRegression(max_iter=100000).fit(extended_train.iloc[:, :-1], extended_train['class'])
    score1 = clf1.score(extended_test, test_classes)
    print('LogisticRegression: ', score1)
    clf2 = RandomForestClassifier(max_depth=100, random_state=0).fit(extended_train.iloc[:, :-1],
                                                                     extended_train['class'])
    score2 = clf2.score(extended_test, test_classes)
    print('RandomForestClassifier: ', score2)

if __name__ == '__main__':
    global fd
    fd = pd.read_csv('train.csv')
    fd.pop('targetName')
    prepare_4_data(1, 4, 7, 10)
    ml(1, 4, 7, 10)
    plt.show()