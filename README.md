# ChargeFlowSim
A simulator of charge flow inside a conductor. A part of 77102 - electricity and magnetism course at Huji - spring semsester 2023. 

Written by Computer Science and Physics students Noam Susman and Dror De-Hartug, 

## project overview
The following code solves 4 different problems presented that were solved analyticaly using mathematical tools and then were solved numerically with the following code. 


## Code overview
Our approach to the problem was to use OOP methods to create a Charge class and to place it in different shapes that represent 2d or 3d space.

 Each problem has a python code file that imports these modules and uses them in different ways to recreate the proper conditions.

 To properly analyize the problem all data is represented visualy during the program run and saved in a spreadsheet when the run is concluded.



## Charge and Shape Classes - Python

This Python project includes two classes, `Charge` and `Shape`, each designed to represent and manipulate different aspects of physical systems involving point charges and geometric shapes.

## Charge Class

The `Charge` class represents a point charge in space and provides methods for simulating its behavior in an electric field.

### Class Overview

The `Charge` class provides the following attributes and methods:

#### Attributes

- `q`: The charge of the point charge in Coulombs (default: -e, the charge of an electron).
- `m`: The mass of the point charge in kilograms (default: mass of an electron).
- `x`, `y`, `z`: The Cartesian coordinates of the point charge in space.
- `efx`, `efy`, `efz`: The components of the electric field acting on the charge.
- `in_sphere`: A flag to indicate whether the charge is inside a sphere.
- `index`: An identifier for the charge.

#### Methods

- `get_position()`: Get the position of the charge as a tuple `(x, y, z)`.
- `get_charge()`: Get the charge of the point charge.
- `set_position(x, y, z)`: Set the position of the charge.
- `set_charge(q)`: Set the charge of the point charge.
- `set_mass(m)`: Set the mass of the point charge.
- `set_in_sphere(in_sphere_value)`: Set the `in_sphere` flag.
- `get_in_sphere()`: Get the value of the `in_sphere` flag.
- `get_radius()`: Calculate and return the radial distance of the charge from the origin.
- `calculate_electric_field(charges, external_field=(0, 0, 0))`: Calculate the electric field acting on the charge due to other charges and an external electric field.
- `update_motion(tao)`: Update the position of the charge under the influence of its electric field.
- `update_position(velocity, tao)`: Move the charge under the effect of a Drude collision velocity.
- `correct_r_to_radius(radius, dim)`: Correct the position of the charge to be at a given radius from the origin in the radial direction.
- `__str__()`: A string representation of the `Charge` object, displaying its properties.

#### Usage

Here's an example of how to create and use a `Charge` object:

```python
from charge import Charge

# Create a charge at coordinates (1.0, 2.0, 3.0) with a charge of 1.0 C and a mass of 9.10938356e-31 kg
charge = Charge(1.0, 2.0, 3.0, index=0, q=1.0, mass=9.10938356e-31)

# Access charge properties
print(f"Charge Position: {charge.get_position()}")
print(f"Charge Charge: {charge.get_charge()} C")
print(f"Charge Mass: {charge.m} kg")

# Calculate the electric field acting on the charge
external_field = (0, 0, 0)
charges = [charge]  # List of charges (only one charge in this example)
charge.calculate_electric_field(charges, external_field)
print(f"Electric Field Components (Ex, Ey, Ez): ({charge.efx}, {charge.efy}, {charge.efz})")

# Update the motion of the charge
tao = 0.01  # Time increment
charge.update_motion(tao)
print(f"Updated Position: {charge.get_position()}")

# Display charge information
print(charge)
```

## Shape Class

The `Shape` class represents geometric shapes and provides methods for manipulating their attributes, such as calculating volume and surface area.

### Class Overview

The `Shape` class provides the following attributes and methods:

#### Attributes

- `x`, `y`, `z`: The coordinates of the center of the shape in three-dimensional space.
- `radius`: The radius of the shape.
- `dimension`: The number of dimensions of the shape (2D or 3D).

#### Methods

- `get_center()`: Get the coordinates of the center of the shape as a tuple `(x, y, z)`.
- `get_radius()`: Get the radius of the shape.
- `get_dimension()`: Get the dimensionality of the shape.
- `set_center(x, y, z)`: Set the coordinates of the center of the shape.
- `set_radius(radius)`: Set the radius of the shape.
- `set_dimension(dimension)`: Set the dimensionality of the shape.
- `calculate_volume()`: Calculate and return the volume of the shape (for 3D shapes only).
- `calculate_area()`: Calculate and return the surface area of the shape (for 2D and 3D shapes).
- `calculate_circumference()`: Calculate and return the circumference of the shape (for 2D shapes only).
- `__str__()`: A string representation of the `Shape` object, displaying its properties.

#### Usage

Here's an example of how to create and use a `Shape` object:

```python
from shape import Shape

# Create a 2D circle at coordinates (1.0, 2.0) with a radius of 3.0 units
circle = Shape(1.0, 2.0, radius=3.0, dimension=2)

# Access shape properties
print(f"Shape Center: {circle.get_center()}")
print(f"Shape Radius: {circle.get_radius()} units")
print(f"Shape Dimension: {circle.get_dimension()}D")

# Calculate and display the circumference of the circle
circumference = circle.calculate_circumference()
print(f"Circumference: {circumference} units")

# Create a 3D sphere at coordinates (0.0, 0.0, 0.0) with a radius of 4.0 units
sphere = Shape(0.0, 0.0, 0.0, radius=4.0, dimension=3)

# Access shape properties
print(f"Shape Center: {sphere.get_center()}")
print(f"Shape Radius: {sphere.get_radius()} units")
print(f"Shape Dimension: {sphere.get_dimension()}D")

# Calculate and display the volume of the sphere
volume = sphere.calculate_volume()
print(f"Volume: {volume} cubic units")

# Display shape information
print(circle)
print(sphere)
```

## Dependencies

Both the `Charge` and `Shape` classes do not have external dependencies and can be used as standalone modules.

---

This README provides an overview of the `Charge` and `Shape` classes and how to use them for simulating point charges and working with geometric shapes. You can customize and extend these classes to suit your specific needs for physical simulations and geometric calculations.

For additional information, consult the code comments and documentation within the respective Python files (`charge.py` and `shape.py`).
