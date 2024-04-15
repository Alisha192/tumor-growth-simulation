import numpy as np
from scipy.stats import norm
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
from Solver import solve_ivp
# Параметры модели
#особенности организма отдельного человека
R = 1.0   #скорость роста опухоли
MU = 0.5  #коэф потребления пит в-в
G = 0.1   #снижение скорости роста из-за огран пит.в-в
K = 0.1   #сила иммун. ответа
D = 0.1   #cнижение иммун. ответа
S = 0.1   #скорость ангиогенеза
H = 0.1   #ингибирование ангиогенеза
N0 = 1    # начальное количество опухолевых клеток
VMAX = 10.0  #макс объём опухоли

# Функция для вычисления правой части системы ОДУ
def f(t, y):
    N, P, I, A, V = y
    #кол-во опух.клеток, концентрация пит.в-в, уровень иммун.ответа, ангиогенез, объём опухоли

# Рост опухоли
    dNdt = R * N - MU * N * P - G * N - K * I * (N - N0)

# Питательные вещества
    dPdt = -R * N * P / MU + D * norm.cdf(P)

# Иммунный ответ
    dIdt = K * I * (N - N0) - D * I

# Ангиогенез
    dAdt = S * N - H * A

# Объем опухоли
    dVdt = R * N * (1 - V / VMAX)

    return np.array([dNdt, dPdt, dIdt, dAdt, dVdt])

#первый метод
# Обновленная функция для отображения графиков
def update_plot(N0, P0, I0, A0, V0):
#в функцию передаются начальные значения: кол-во опух.клеток, концентрация пит.в-в, уровень иммун.ответа, ангиогенез, объём опухоли
# Решение системы ОДУ с адаптивным методом Рунге-Кутты
    sol = solve_ivp(f, (0.0, 100.0), np.array([N0, P0, I0, A0, V0]), method="RK45", atol=1e-6, rtol=1e-3)
# Отрисовка графиков
    plt.figure(figsize=(8, 6))
    label_dict = {0: "Рост опухоли", 1: "Питательные вещества", 2: "Иммунный ответ", 3: "Ангиогенез", 4: "Объем опухоли"}



    for i, var in enumerate(sol.y):
        plt.plot(sol.t, var, label=f"y{i + 1} - {label_dict[i]}")

# Подписи к осям
    plt.xlabel("Время (в условных единицах)")
    plt.ylabel("Значение")

# Подпись к легенде
    plt.legend(loc="best")

# Заголовок графика
    plt.title(f"\nN0={N0:.2f}, P0={P0:.2f}, I0={I0:.2f}, A0={A0:.2f}, V0={V0:.2f}")

    plt.show()

#второй метод
def update_plot2(N0, P0, I0, A0, V0):
    # в функцию передаются начальные значения: кол-во опух.клеток, концентрация пит.в-в, уровень иммун.ответа, ангиогенез, объём опухоли
    # Решение системы ОДУ с адаптивным методом Рунге-Кутты
    sol = solve_ivp(f, (0.0, 100.0), np.array([N0, P0, I0, A0, V0]), method="RK23", atol=1e-6, rtol=1e-3)
    # Отрисовка графиков
    plt.figure(figsize=(8, 6))
    label_dict = {0: "Рост опухоли", 1: "Питательные вещества", 2: "Иммунный ответ", 3: "Ангиогенез",
                  4: "Объем опухоли"}

    for i, var in enumerate(sol.y):
        plt.plot(sol.t, var, label=f"y{i + 1} - {label_dict[i]}")

    # Подписи к осям
    plt.xlabel("Время (в условных единицах)")
    plt.ylabel("Значение")

    # Подпись к легенде
    plt.legend(loc="best")

    # Заголовок графика
    plt.title(f"\nN0={N0:.2f}, P0={P0:.2f}, I0={I0:.2f}, A0={A0:.2f}, V0={V0:.2f}")

    plt.show()


# Виджеты для параметров
N0_slider = widgets.FloatSlider(min=0.01, max=10.0, value=1.0, step=0.01)
P0_slider = widgets.FloatSlider(min=0.01, max=10.0, value=1.0, step=0.01)
I0_slider = widgets.FloatSlider(min=0.01, max=10.0, value=0.1, step=0.01)
A0_slider = widgets.FloatSlider(min=0.01, max=10.0, value=0.1, step=0.01)
V0_slider = widgets.FloatSlider(min=0.01, max=10.0, value=0.1, step=0.01)

# Обновление графика при изменении параметров
out = widgets.interactive_output(update_plot, {'N0': N0_slider, 'P0': P0_slider, 'I0': I0_slider, 'A0': A0_slider, 'V0': V0_slider})
ut = widgets.interactive_output(update_plot2, {'N0': N0_slider, 'P0': P0_slider, 'I0': I0_slider, 'A0': A0_slider, 'V0': V0_slider})

# Отображение виджетов и графика
display(N0_slider, P0_slider, I0_slider, A0_slider, V0_slider, out)
display(N0_slider, P0_slider, I0_slider, A0_slider, V0_slider, ut)
