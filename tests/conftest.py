import sys


def pytest_collection_finish(session):
    print("\nMODULES models:", sorted(m for m in sys.modules if m.endswith("models")))
