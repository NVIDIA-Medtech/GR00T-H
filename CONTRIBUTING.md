# Contributing to GR00T-H

Thanks for considering contributing to GR00T-H! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

> **Note:** GR00T-H is a healthcare-focused fork of [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T). For the latest on the base GR00T model, general robotics features, and non-healthcare contributions, please refer to the upstream [Isaac-GR00T repository](https://github.com/NVIDIA/Isaac-GR00T). Contributions specific to healthcare robotics, the Open-H dataset, or surgical embodiments belong here.

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

## Bug reports and feature requests

### Did you find a bug?

First, determine whether the issue is specific to GR00T-H (healthcare embodiments, Open-H dataset, surgical data pipeline) or affects the base GR00T model:

- **GR00T-H specific issues**: Search [GR00T-H issues](https://github.com/NVIDIA-Medtech/GR00T-H/issues), then [open a new issue](https://github.com/NVIDIA-Medtech/GR00T-H/issues/new) if not already reported.
- **Base GR00T issues**: Report on the upstream [Isaac-GR00T issues](https://github.com/NVIDIA/Isaac-GR00T/issues).

Be sure to include a clear title and description with as much relevant information as possible, including how to reproduce the issue and the behavior you expect to see.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create a feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing it first on a GitHub issue.
* Check the documentation to make sure your feature does not already exist.
* Do [a quick search](https://github.com/NVIDIA-Medtech/GR00T-H/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/GR00T-H.git

    or

        git clone git@github.com:USERNAME/GR00T-H.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/GR00T-H.git, but doesn't know anything the *main* repo, [https://github.com/NVIDIA-Medtech/GR00T-H.git](https://github.com/NVIDIA-Medtech/GR00T-H). You can see this by running

        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/GR00T-H.git (fetch)
        origin https://github.com/USERNAME/GR00T-H.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/NVIDIA-Medtech/GR00T-H.git](https://github.com/NVIDIA-Medtech/GR00T-H). To do this, run the following:

        git remote add upstream https://github.com/NVIDIA-Medtech/GR00T-H.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/GR00T-H.git (fetch)
        origin https://github.com/USERNAME/GR00T-H.git (push)
        upstream https://github.com/NVIDIA-Medtech/GR00T-H.git (fetch)
        upstream https://github.com/NVIDIA-Medtech/GR00T-H.git (push)

    Finally, you'll need to create a Python 3 virtual environment suitable for working on this project.
    ```bash
    uv pip install -e .[dev]
    ```

    The "editable mode" comes from the `-e` argument to `pip`, and essentially just creates a symbolic link from the site-packages directory of your virtual environment to the source code in your local clone. That way any changes you make will be immediately reflected in your virtual environment.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/NVIDIA-Medtech/GR00T-H.git](https://github.com/NVIDIA-Medtech/GR00T-H), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Test your changes**

    <details><summary>Expand details 👇</summary><br/>

    First, you should run [`ruff`](https://docs.astral.sh/ruff/) to make sure your code is formatted consistently:

    ```bash
    ruff format .
    ruff check --fix .
    ```

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/NVIDIA-Medtech/GR00T-H/tree/main/tests). These tests are run with [`pytest`](https://docs.pytest.org/en/latest/), which you can use to locally run any test modules that you've added or changed.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/NVIDIA-Medtech/GR00T-H/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

## Full text of the DCO

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
