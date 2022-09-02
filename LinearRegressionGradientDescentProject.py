from numpy import *


def compute_error_for_line_given_points(b, m, points):
    # Initialize error at 0
    total_error = 0
    # For every point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # Starting points
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m
        b_gradient += -(2 / n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + b_current))
    # Update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    # Gradient descent
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    # Collect our data:
    points = genfromtxt('data.csv', delimiter=',')
    # Defining hyper parameters:
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 10000
    # Train our model:
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('After {0} iterations b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m,
                                                                      compute_error_for_line_given_points(b, m,
                                                                                                          points)))


if __name__ == '__main__':
    run()
