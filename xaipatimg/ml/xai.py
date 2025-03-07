
def generate_cam_resnet18(cam_technique, db_dir, model_path, dataset_filename):

    os.makedirs(img_dir, exist_ok=True)
    target_layers = [model.layer4[-1]]

    input_tensor = torch.from_numpy(X)
    input_tensor.to(device)
    print(input_tensor.shape)
    print(y.shape)
    # We have to specify the target we want to generate the CAM for.
    for i in range(N_generated):
        # Construct the CAM object once, and then re-use it on many images.
        for j, targets in enumerate([[ClassifierOutputTarget(0)], [ClassifierOutputTarget(1)]]):
            with cam_type(model=model, target_layers=target_layers) as cam:
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=input_tensor[i:1 + i], targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                bgr_img = cv2.imread(paths_list[i], 1)
                bgr_img = cv2.resize(bgr_img, (256, 256))
                cv2.imwrite(img_dir + str(i) + "_class" + str(y[i]) + ".jpg", bgr_img)
                bgr_img = np.float32(bgr_img) / 255

                visualization = show_cam_on_image(bgr_img, grayscale_cam[0], use_rgb=False)
                probabilities = torch.nn.functional.softmax(cam.outputs, dim=1)
                cv2.imwrite(img_dir + str(i) + "_class" + str(y[i]) + "_pred" + str(
                    probabilities.round().int().T[1].cpu().numpy()) +
                            "_gradcamtarget" + str(j) + ".jpg", visualization)