# Submit Your Policy

- Obtain Credentials: You can obtain your Docker login credentials by clicking the "Get Registry Token" button on the My Submissions page of the Test Server. (Note that the validity period of the voucher is 1 hour)

- Build the Image: Integrate all necessary components—including dependencies, model checkpoints, and inference scripts—into a single Docker image.

- Configure Port and Entrypoint:

    - Port Requirement: Your inference service must be configured to listen on port 8999.
    - Startup: Ensure the Docker image is configured to start the inference container directly via an ENTRYPOINT or CMD instruction, without requiring any additional commands at runtime.

- Tag and Push: Tag your image according to the registry endpoint and namespace provided. For example, if your registry endpoint is registry.test.com and your namespace is test1, your image should be tagged and pushed as follows:

    - Tag format: registry.test.com/test1/{image_name}:tag
    - Command: docker push registry.test.com/test1/{image_name}:tag

- Commands for this repository (run from the `ACoT-VLA` directory):

    Replace `<your-image>` with your image name. Build with the **full registry name** so you do not need a separate `docker tag`:

    ```bash
    docker build -f scripts/docker/serve_policy.Dockerfile \
      -t sim-icra-registry.cn-beijing.cr.aliyuncs.com/fvlmotio/<your-image>:latest .

    docker login sim-icra-registry.cn-beijing.cr.aliyuncs.com

    docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/fvlmotio/<your-image>:latest
    ```

    If you already built under another local name only, retag then push:

    ```bash
    docker tag <your-image>:latest sim-icra-registry.cn-beijing.cr.aliyuncs.com/fvlmotio/<your-image>:latest
    docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/fvlmotio/<your-image>:latest
    ```

    Note: Files listed in `.dockerignore` are not copied into the image; ensure checkpoints and any other required assets are included in the build context.

- Specify Model Type: Choose your model type abs_joint or abs_pose on the submission page according to your model.

- Troubleshooting & Optimization

    - Credential Expiration: If the image is too large to complete the upload within the credential's validity period, simply refresh your login credentials and re-initiate the push. Docker's layer-based architecture ensures that previously successful layers are preserved; only the remaining layers will be transmitted.
    - Image Optimization: If the push continues to fail, please optimize the Docker image to reduce its footprint. We recommend:
        - Cleaning up builds caches and unnecessary dependencies.
        - Implementing multi-stage builds to minimize the final image size.