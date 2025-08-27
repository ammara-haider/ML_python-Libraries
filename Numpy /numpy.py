import numpy as np

# -----------------------------
# 🔹 1. Creating Arrays
# -----------------------------
# From list
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# 2D array
arr2d = np.array([[1,2,3],[4,5,6]])
print("2D Array:\n", arr2d)

# Predefined arrays
print("Zeros:\n", np.zeros((3,3)))
print("Ones:\n", np.ones((2,2)))
print("Identity Matrix:\n", np.eye(3))
print("Range:", np.arange(0,10,2))
print("Linspace:", np.linspace(0,1,5))

# -----------------------------
# 🔹 2. Array Attributes
# -----------------------------
arr = np.array([[1,2,3],[4,5,6]])
print("Dimensions:", arr.ndim)
print("Shape:", arr.shape)
print("Size:", arr.size)
print("Datatype:", arr.dtype)

# -----------------------------
# 🔹 3. Indexing & Slicing
# -----------------------------
arr = np.array([10,20,30,40,50])
print("First element:", arr[0])
print("Last element:", arr[-1])
print("Slice [1:4]:", arr[1:4])

arr2d = np.array([[1,2,3],[4,5,6]])
print("Row 0, Col 1:", arr2d[0,1])
print("All rows, col 1:", arr2d[:,1])

# -----------------------------
# 🔹 4. Array Operations
# -----------------------------
a = np.array([1,2,3])
b = np.array([4,5,6])

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Multiplication:", a*b)
print("Division:", a/b)

# Broadcasting
m = np.array([[1],[2],[3]])
n = np.array([10,20,30])
print("Broadcasting:\n", m+n)

# -----------------------------
# 🔹 5. Universal Functions
# -----------------------------
arr = np.array([1,4,9,16])

print("Square Root:", np.sqrt(arr))
print("Exponential:", np.exp(arr))
print("Log:", np.log(arr))
print("Sine:", np.sin(arr))
print("Max:", np.max(arr))
print("Mean:", np.mean(arr))
print("Std:", np.std(arr))

# -----------------------------
# 🔹 6. Reshaping Arrays
# -----------------------------
arr = np.arange(12)
reshaped = arr.reshape(3,4)
print("Reshaped:\n", reshaped)

flat = reshaped.flatten()
print("Flattened:", flat)

# -----------------------------
# 🔹 7. Stacking Arrays
# -----------------------------
a = np.array([1,2,3])
b = np.array([4,5,6])

print("Vertical Stack:\n", np.vstack((a,b)))
print("Horizontal Stack:\n", np.hstack((a,b)))
print("Column Stack:\n", np.column_stack((a,b)))

# -----------------------------
# 🔹 8. Boolean Indexing
# -----------------------------
arr = np.array([10,20,30,40,50])
print("Condition:", arr > 25)
print("Filtered:", arr[arr > 25])

# -----------------------------
# 🔹 9. Random Numbers
# -----------------------------
np.random.seed(42)

print("Random floats:", np.random.rand(3))
print("Random ints:", np.random.randint(1,10,5))
print("Normal distribution:\n", np.random.randn(3,3))
print("Random choice:", np.random.choice([1,2,3,4,5], size=3))

# -----------------------------
# 🔹 10. Linear Algebra
# -----------------------------
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print("Matrix Multiplication:\n", np.dot(A,B))
print("Determinant:", np.linalg.det(A))
print("Inverse:\n", np.linalg.inv(A))
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# -----------------------------
# 🔹 11. Useful Functions
# -----------------------------
arr = np.array([1,2,3,4,5])

print("Unique:", np.unique(arr))
print("Sorted:", np.sort(arr))
print("Indices sorted:", np.argsort(arr))
print("Where >2:", np.where(arr>2))

# -----------------------------
# 🎯 Summary
# -----------------------------
print("""
NumPy is the foundation of ML
Key areas:
- Arrays (creation, indexing, slicing)
- Operations & broadcasting
- Statistics & math functions
- Linear algebra
- Random numbers
""")
