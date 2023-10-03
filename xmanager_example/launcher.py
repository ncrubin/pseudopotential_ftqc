# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl import app, flags
from xmanager import xm, xm_abc
from xmanager.contrib import gcs

flags.adopt_module_key_flags(gcs)  # --xm_gcs_path, need this for cloud bucket location

# should match the name of the directoryin which all your xmanager script live
_package_name = "lambda_nonloc"
# Default image comes with python3.7
_DEFAULT_STEPS = rf"""ENV LANG=C.UTF-8
FROM python:3.9
COPY {_package_name}/requirements.txt /{_package_name}/requirements.txt
RUN python -m pip install -r {_package_name}/requirements.txt
COPY {_package_name}/ /{_package_name}
RUN python -m pip install -r {_package_name}/pseudopotential_ftqc/requirements.txt
RUN python -m pip install {_package_name}/pseudopotential_ftqc
RUN chown -R 1000:root /{_package_name} && chmod -R 775 /{_package_name}
WORKDIR {_package_name}
"""


def main(_):
    title = "lambda_nonloc"
    num_cpu = 1
    with xm_abc.create_experiment(experiment_title=title) as experiment:
        job_requirements = xm.JobRequirements(cpu=num_cpu, memory=2 * num_cpu * xm.GiB)
        executor = xm_abc.Gcp(requirements=job_requirements)
        spec = xm.PythonContainer(
            # Package the current directory that this script is in.
            path=".",
            # Docker instructions from above
            docker_instructions=_DEFAULT_STEPS.split("\n"),
            # This is the command which will run inside our container
            entrypoint=xm.CommandList([f"export OMP_NUM_THREADS={num_cpu}; export MKL_NUM_THREADS={num_cpu}; python -m run_job"]),
        )
        workdir = gcs.get_gcs_path_or_fail(title)
        workdir = gcs.get_gcs_fuse_path(workdir)
        [executable] = experiment.package(
            [
                xm.Packageable(
                    executable_spec=spec,
                    executor_spec=executor.Spec(),
                    args={
                        "output_dir": workdir + "/lambda_nonloc",
                        "nut": 35,
                        "n1": 4,
                        "n2": 5,
                        "n3": 5,
                        "lattice": 5,
                        "atom": "Pd",
                        },
                ),
            ]
        )
        experiment.add(
            xm.Job(
                executable=executable,
                executor=executor,
                name="test-job",
            ),
        )


if __name__ == "__main__":
    app.run(main)
