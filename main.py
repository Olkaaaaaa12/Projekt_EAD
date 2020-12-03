import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sqlite3

def zadanie1():
    names = pd.DataFrame(pd.read_csv("names/yob1880.txt", names=["Name", "Sex", "Val"]))
    names["Year"] = 1880

    for i in range(1881, 2020):
        file = pd.DataFrame(pd.read_csv("names/yob" + str(i) + ".txt", names=["Name", "Sex", "Val"]))
        file["Year"] = i
        names = pd.concat([names, file], ignore_index=True)

    return names

def zadanie2(names):
    zad2 = names.pivot_table(index="Name", columns="Sex", values="Val", aggfunc=np.sum, fill_value=0)
    print("Zadanie 2: W tym czasie zostało nadanych", len(zad2), "unikalnych imion.")

    return zad2

def zadanie3(zad2):
    n = list(zad2.index)
    wom = 0
    men = 0
    for l in n:
        if zad2["F"][l] > 0:
            wom += 1
        if zad2["M"][l] > 0:
            men += 1
    print("Zadanie 3: Nadano", wom, "unikalnych żeńskich imion i", men, "unikalnych męskich imion.")

def zadanie4(names):
    zad4 = names.pivot_table(index="Year", columns="Sex", values="Val", aggfunc=np.sum, fill_value=0)
    names["Frequency_male"] = 0
    names["Frequency_female"] = 0
    temp = names[names.Year.isin([1880]) & names.Sex.isin(["F"])]
    temp.loc[:, "Frequency_female"] = temp["Val"] / zad4["F"][1880]
    temp2 = names[names.Year.isin([1880]) & names.Sex.isin(["M"])]
    temp2.loc[:, "Frequency_male"] = temp2["Val"] / zad4["M"][1880]
    new = pd.concat([temp, temp2])
    for y in range(1881, 2020):
        for s in ["F", "M"]:
            if s == "F":
                temp = names[names.Year.isin([y]) & names.Sex.isin([s])]
                temp.loc[:, "Frequency_female"] = temp["Val"] / zad4[s][y]
            else:
                temp2 = names[names.Year.isin([y]) & names.Sex.isin([s])]
                temp2.loc[:, "Frequency_male"] = temp2["Val"] / zad4[s][y]
        new = pd.concat([new, temp, temp2], ignore_index=True)
    return new, zad4

def zadanie5(zad4):
    birthY = []
    birthS = []
    for y in zad4.index:
        birthY.append(sum(zad4.loc[y, :]))
        birthS.append(zad4.loc[y, "F"] / zad4.loc[y, "M"])
    fig, axs = plt.subplots(2)
    axs[0].plot(zad4.index, birthY)
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Amount of births")
    axs[1].plot(zad4.index, birthS)
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Ratio of girl's births to boy's births")
    yearList = list(zad4.index)
    for i in range(len(birthS)):
        birthS[i] = abs(birthS[i] - 1)
    ymax = max(birthS)
    xpos = birthS.index(ymax)
    xmax = yearList[xpos]
    ymin = min(birthS)
    xposm = birthS.index(ymin)
    xmin = yearList[xposm]
    fig.suptitle('Zadanie 5')
    print("Zadanie 5: Najmniejszą różnicę zanotowano w " + str(xmin) + ", a największą w " + str(xmax) + ".")

    return birthY

def zadanie6(names):
    zad6Wf = names[names["Sex"] == "F"]
    zad6Mf = names[names["Sex"] == "M"]
    zad6W = zad6Wf.pivot_table(index=["Year", "Name"], values=["Val"])
    zad6W.sort_values(by=["Year", "Val"], inplace=True, ascending=False)
    zad6M = zad6Mf.pivot_table(index=["Year", "Name"], values=["Val"])
    zad6M.sort_values(by=["Year", "Val"], inplace=True, ascending=False)
    valM = []
    valMall = []
    valWall = []
    valW = []
    maxW = zad6W[zad6W.index.isin([1880], level=0)]
    valWall.append(len(maxW))
    maxW = maxW.head(1000)
    valW.append(len(maxW))
    maxM = zad6M[zad6M.index.isin([1880], level=0)]
    valMall.append(len(maxM))
    maxM = maxM.head(1000)
    valM.append(len(maxM))
    for i in range(1881, 2020):
        mW = zad6W[zad6W.index.isin([i], level=0)]
        valWall.append(len(mW))
        mW = mW.head(1000)
        valW.append(len(mW))
        maxW = pd.concat([maxW, mW])
        mM = zad6M[zad6M.index.isin([i], level=0)]
        valMall.append(len(mM))
        mM = mM.head(1000)
        valM.append(len(mM))
        maxM = pd.concat([maxM, mM])
    maxM = maxM.pivot_table(index=["Name"], values=["Val"], aggfunc=np.sum)
    maxW = maxW.pivot_table(index=["Name"], values=["Val"], aggfunc=np.sum)
    maxW.sort_values(by=["Val"], inplace=True, ascending=False)
    maxM.sort_values(by=["Val"], inplace=True, ascending=False)
    maxM = maxM.head(1000)
    maxW = maxW.head(1000)
    print("Zadanie 6: Top1000 żeńskich: \n", maxW)
    print("Zadanie 6: Top1000 męskich: \n", maxM)

    return maxM, maxW, valMall, valWall, mM,  mW, valM, valW

def zadanie7(maxM, maxW, names, zad4, birthY):
    bestM = maxM.index[0]
    bestW = maxW.index[0]
    zad7 = names.pivot_table(index=["Name"], columns=["Year"], values=["Val"], fill_value=0, aggfunc=np.sum)
    zad7f = zad7[zad7.index.isin([bestW, bestM, "Harry", "Marilin"])]
    harry = []
    bM = []
    bW = []
    marilin = []
    listName = [harry, bM, bW, marilin]
    nam = ['Harry', str(bestM), str(bestW), 'Marilin']
    for i in range(len(zad4.index)):
        for ind in range(len(listName)):
            val = (zad7f.loc[nam[ind], ("Val", zad4.index[i])]) / birthY[i]
            listName[ind].append(val)
    print("Zadanie 7: W latach 1940, 1980 i 2019 imie Harry nadano kolejno " + str(zad7f.loc['Harry', ("Val", 1940)]) + ", " + \
          str(zad7f.loc['Harry', ("Val", 1980)]) + ", " + str(
        zad7f.loc['Harry', ("Val", 2019)]) + " razy, imie Marilin " + str(zad7f.loc['Marilin', ("Val", 1940)]) + \
          ", " + str(zad7f.loc['Marilin', ("Val", 1980)]) + ", " + str(
        zad7f.loc['Marilin', ("Val", 2019)]) + " razy, imie " + str(bestW) + " " + str(
        zad7f.loc[str(bestW), ("Val", 1940)]) + \
          ", " + str(zad7f.loc[str(bestW), ("Val", 1980)]) + ", " + str(
        zad7f.loc[str(bestW), ("Val", 2019)]) + " razy, a imie " + str(bestM) + " " + str(
        zad7f.loc[str(bestM), ("Val", 1940)]) + \
          ", " + str(zad7f.loc[str(bestM), ("Val", 1980)]) + ", " + str(
        zad7f.loc[str(bestM), ("Val", 2019)]) + " razy.")
    fig2, axs2 = plt.subplots()
    axs2.plot(zad4.index, list(zad7f.loc["Harry", :]), '-b')
    axs2.plot(zad4.index, list(zad7f.loc["Marilin", :]), '--b')
    axs2.plot(zad4.index, list(zad7f.loc[str(bestM), :]),'-.b')
    axs2.plot(zad4.index, list(zad7f.loc[str(bestW), :]), ':b')
    axs2.legend(['Harry', 'Marilin', str(bestM), str(bestW)])
    axs2.set_xlabel("Year")
    axs2.set_ylabel("Amount", color='b')
    axs2.tick_params(axis='y', labelcolor='b')
    axs22 = axs2.twinx()
    axs22.set_ylabel("Frequency", color='r')
    axs22.tick_params(axis='y', labelcolor='r')
    axs22.plot(zad4.index, harry, '-r')
    axs22.plot(zad4.index, marilin, '--r')
    axs22.plot(zad4.index, bM, '-.r')
    axs22.plot(zad4.index, bW, ':r')
    fig2.tight_layout()
    fig2.suptitle('Zadanie 7')

def zadanie8(valM, valW, valMall, valWall, zad4):
    zad8M = []
    zad8W = []
    for i in range(len(valMall)):
        zad8M.append((valM[i] / valMall[i]) * 100)
        zad8W.append((valW[i] / valWall[i]) * 100)
    maxD = 0
    ids = 0
    for i in range(len(zad8M)):
        sc = abs(zad8M[i] - zad8W[i])
        if sc > maxD:
            maxD = sc
            ids = i
    print("Zadanie 8: Najwieksza roznice odnotowano w", zad4.index[ids], "roku.")
    fig3, axs3 = plt.subplots()
    axs3.plot(zad4.index, zad8M)
    axs3.plot(zad4.index, zad8W)
    axs3.set_xlabel("Year")
    axs3.set_ylabel("Percent of names from top1000")
    axs3.legend(['Men', 'Women'])
    fig3.suptitle('Zadanie 8')

def zadanie9(names, zad4):
    namListall = list(names['Name'])
    for i in range(len(namListall)):
        namListall[i] = namListall[i][-1]

    names["Last_letter"] = namListall
    zad9 = names.pivot_table(index=["Year", "Sex", "Last_letter"], values="Val", aggfunc=np.sum, fill_value=0)
    schf = zad9[zad9.index.get_level_values("Year").isin(['1880'])]
    temp = schf[schf.index.get_level_values("Sex").isin(["M"])]
    temp.loc[:, "Val"] = temp["Val"] / zad4["M"][1880]
    temp2 = schf[schf.index.get_level_values("Sex").isin(["F"])]
    temp2.loc[:, "Val"] = temp2["Val"] / zad4["F"][1880]
    temp = pd.concat([temp, temp2])
    for i in range(1881, 2020):
        sch = zad9[zad9.index.get_level_values("Year").isin([str(i)])]
        for s in ["M", "F"]:
            temp2 = sch[sch.index.get_level_values("Sex").isin([s])]
            temp2.loc[:, "Val"] = temp2["Val"] / zad4[s][i]
            temp = pd.concat([temp, temp2])
    zad9f = temp[temp.index.get_level_values("Year").isin(["1910", "1960", "2015"])]
    bar = zad9f.pivot_table(index="Last_letter", columns=["Year", "Sex"], fill_value=0)
    zad9s = temp.pivot_table(index="Last_letter", columns=["Year", "Sex"], fill_value=0)
    labels = list(bar.index)
    minL = []

    for i in labels:
        val = abs(bar[("Val", 1910, "M")][i] - bar[("Val", 2015, "M")][i])
        minL.append(val)

    Tdf = pd.DataFrame(index=labels, data=minL, columns=["Val"])
    Tdf.sort_values(by=["Val"], inplace=True, ascending=False)
    print("Zadanie 9: Najwiekszy wzrost/spadek miedzy 1910 a 2015 nastapil dla litery " + str(Tdf.index[0]) + ".")
    fig, ax = plt.subplots()

    cosW = list(bar[("Val", 1910, "F")])
    cos2W = list(bar[("Val", 1960, "F")])
    cos3W = list(bar[("Val", 2015, "F")])

    cosM = list(bar[("Val", 1910, "M")])
    cos2M = list(bar[("Val", 1960, "M")])
    cos3M = list(bar[("Val", 2015, "M")])

    x = np.arange(len(cosW))
    width = 0.13
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelrotation=0)
    ax.bar(x - 2.5 * width, cosW, width, label='1910 women')
    ax.bar(x - 1.5 * width, cosM, width, label='1910 men')
    ax.bar(x - 0.5 * width, cos2W, width, label='1960 women')
    ax.bar(x + 0.5 * width, cos2M, width, label='1960 men')
    ax.bar(x + 1.5 * width, cos3W, width, label='2015 women')
    ax.bar(x + 2.5 * width, cos3M, width, label='2015 men')
    ax.legend()
    fig.suptitle('Zadanie 9')

    first3 = list(Tdf.index)[0:3]
    one = []
    two = []
    three = []
    for i in range(1880, 2020):
        for ind, j in enumerate(first3):
            if ind == 0:
                one.append(zad9s[("Val", i, "M")][j])
            elif ind == 1:
                two.append(zad9s[("Val", i, "M")][j])
            else:
                three.append(zad9s[("Val", i, "M")][j])

    fig, axs4 = plt.subplots()
    axs4.plot(zad4.index, one)
    axs4.plot(zad4.index, two)
    axs4.plot(zad4.index, three)
    axs4.set_xlabel("Year")
    axs4.set_ylabel("Popularity of the last letter")
    axs4.legend([str(first3[0]), str(first3[1]), str(first3[2])])
    fig.suptitle('Zadanie 9')

def zadanie10(zad2, names):
    mix = []
    for l in list(zad2.index):
        if zad2["F"][l] > 0 and zad2["M"][l] > 0:
            mix.append(l)
    zad10 = names[names.Name.isin(mix)]
    zad10f = zad10.pivot_table(index="Name", columns="Sex", values="Val", aggfunc=np.sum)
    zad10W = zad10f.sort_values(by=["F"], inplace=False, ascending=False)
    zad10M = zad10f.sort_values(by=["M"], inplace=False, ascending=False)
    print("Zadanie 10: Najpopularniejsze imie żeńskie to " + str(zad10W.index[0]) + ", natomiast męskie to " + str(
        zad10M.index[0]) + ".")

    return zad10

def zadanie11(zad10, zad4):
    zad11sum = zad10.pivot_table(index="Name", columns="Year", values="Val", aggfunc=np.sum, fill_value=0)
    zad11prop = zad10.pivot_table(index="Name", columns=["Year", "Sex"], values="Val", fill_value=0)
    for i in range(1880, 2020):
        for s in ["F", "M"]:
            zad11prop.loc[:, (i, s)] = zad11prop[(i, s)] / zad11sum[i]
    ind1 = list(zad4.index).index(1880)
    ind2 = list(zad4.index).index(1920)
    ind3 = list(zad4.index).index(2000)
    ind4 = list(zad4.index).index(2019)
    zad11f = zad11prop.iloc[:, ind1:(ind2 + 1) * 2]
    zad11s = zad11prop.iloc[:, ind3 * 2:(ind4 + 1) * 2]
    zad11s = zad11s.mean(axis=1, level=1)
    zad11f = zad11f.mean(axis=1, level=1)
    sch = pd.concat([zad11f, zad11s], axis=1)
    sch.dropna(inplace=True)
    zad11f = sch.iloc[:, 0:2]
    zad11s = sch.iloc[:, 2:4]
    zad11f["Difference"] = zad11f["F"] - zad11f["M"]
    zad11s["Difference"] = zad11s["F"] - zad11s["M"]
    diff = list(abs(zad11s["Difference"] - zad11f["Difference"]))
    sch["Difference"] = diff
    sch.sort_values(by=["Difference"], inplace=True, ascending=False)
    sch = sch[sch.Difference.isin([2.0])]
    sch.sort_values(by=["Name"], inplace=True)
    print("Zadanie 11: Największa różnica w nadawaniu danego imienia innej płci po pewnym czasie została zanotowana dla imion",
          sch.index[0], "i", sch.index[1], ".")

    zad11prop = zad11prop.droplevel(0, axis=1)
    w = zad11prop.iloc[:, 0::2]
    m = zad11prop.iloc[:, 1::2]
    fig, axs5 = plt.subplots(2)
    axs5[0].plot(zad4.index, w.loc[str(sch.index[0]), :], '.')
    axs5[0].plot(zad4.index, m.loc[str(sch.index[0]), :], '.')
    axs5[1].plot(zad4.index, w.loc[str(sch.index[1]), :], '.')
    axs5[1].plot(zad4.index, m.loc[str(sch.index[1]), :], '.')
    axs5[0].set_xlabel("Year")
    axs5[1].set_xlabel("Year")
    axs5[0].set_ylabel("Birth ratio")
    axs5[1].set_ylabel("Birth ratio")
    axs5[0].legend(["Women", "Men"], loc='center right')
    axs5[1].legend(["Women", "Men"], loc='center right')
    axs5[0].set_title(str(sch.index[0]), fontsize=16)
    axs5[1].set_title(str(sch.index[1]), fontsize=16)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('Zadanie 11')

def zadanie12():
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()

    c.execute(
        'CREATE TABLE new_table AS SELECT * FROM USA_mltper_1x1 UNION ALL SELECT * FROM USA_fltper_1x1 ORDER BY Year, Sex')
    conn.commit()
    conn.close()

def zadanie13(zad4, birthY):
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()
    deaths = []

    for y in range(1959, 2018):
        c.execute('SELECT sum(dx) FROM new_table WHERE Year =?', (y,))
        deaths.append(c.fetchone()[0])

    prz = []
    ind1 = list(zad4.index).index(1959)
    ind2 = list(zad4.index).index(2018)
    for i in range(ind1, ind2):
        prz.append(birthY[i] - deaths[i - ind1])
    fig, axs = plt.subplots()
    axs.plot(range(1959, 2018), prz)
    axs.set_xlabel("Year")
    axs.set_ylabel("Birthrate")
    fig.suptitle('Zadanie 13')
    conn.close()

def zadanie14(zad4, birthY):
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()
    fyearDeaths = []
    for i in range(1959, 2018):
        c.execute('SELECT sum(dx) FROM new_table WHERE Year=? AND Age=0', (i,))
        fyearDeaths.append(c.fetchone()[0])
    wsp = []
    ind1 = list(zad4.index).index(1959)
    ind2 = list(zad4.index).index(2018)
    for i in range(ind1, ind2):
        wsp.append((birthY[i] - fyearDeaths[i - ind1]) / birthY[i])
    fig, axs = plt.subplots()
    axs.plot(range(1959, 2018), wsp)
    axs.set_xlabel("Year")
    axs.set_ylabel("Survival rate")
    fig.suptitle('Zadanie 14')
    conn.close()
    return wsp

def zadanie15(wsp, zad4, birthY):
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    c = conn.cursor()
    year5Deaths = []

    for i in range(1959, 2013):
        sch = 0
        for j in range(0, 4):
            c.execute('SELECT sum(dx) FROM new_table WHERE Year=? AND Age=?', (i + j, j))
            sch += c.fetchone()[0]
        year5Deaths.append(sch)
    wsp2 = []
    ind1 = list(zad4.index).index(1959)
    ind2 = list(zad4.index).index(2013)
    for i in range(ind1, ind2):
        wsp2.append((birthY[i] - year5Deaths[i - ind1]) / birthY[i])
    fig, axs = plt.subplots()
    axs.plot(range(1959, 2018), wsp)
    axs.plot(range(1959, 2013), wsp2)
    axs.set_xlabel("Year")
    axs.set_ylabel("Survival rate")
    axs.legend(['Survived in first year', 'Survived in first 5 years'])
    fig.suptitle('Zadanie 15')
    c.execute('DROP TABLE new_table')
    conn.close()

if __name__=='__main__':
    names = zadanie1()

    zad2 = zadanie2(names)

    zadanie3(zad2)

    names, zad4 = zadanie4(names)

    birthY = zadanie5(zad4)

    maxM, maxW, valMall, valWall, mM, mW, valM, valW = zadanie6(names)

    zadanie7(maxM, maxW, names, zad4, birthY)

    zadanie8(valM, valW, valMall, valWall, zad4)

    zadanie9(names, zad4)

    zad10 = zadanie10(zad2, names)

    zadanie11(zad10, zad4)

    zadanie12()

    zadanie13(zad4, birthY)

    wsp = zadanie14(zad4, birthY)

    zadanie15(wsp, zad4, birthY)

    plt.show()

