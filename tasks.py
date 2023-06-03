from invoke import task


@task
def verify(c):
    print("Running black...")
    c.run("poetry run black .")
    print("Running ruff...")
    c.run("poetry run ruff . --fix")
    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Running pytest...")
    c.run("poetry run pytest")
