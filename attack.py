import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import feature_squeezing as fs


def ad_attack(file_name, file_path, epsilon):
    mpl.rcParams['figure.figsize'] = (8, 8)
    mpl.rcParams['axes.grid'] = False

    # load pretrained MobileNetV2 model
    pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                         weights='imagenet')
    pretrained_model.trainable = False

    # ImageNet labels
    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

    # Feature squeezing detector
    # TODO: make 3 squeezers
    squeezers_all = ['bit_depth_5', 'median_filter_2_2', 'non_local_means_color_11_3_4']
    squeezers_bit = ['bit_depth_5']
    squeezers_median = ['median_filter_2_2']
    squeezers_nlocal = ['non_local_means_color_11_3_4']

    detector_all = fs.FeatureSqueezingDetector(pretrained_model, squeezers_all)
    detector_bit = fs.FeatureSqueezingDetector(pretrained_model, squeezers_bit)
    detector_median = fs.FeatureSqueezingDetector(pretrained_model, squeezers_median)
    detector_nlocal = fs.FeatureSqueezingDetector(pretrained_model, squeezers_nlocal)

    # Helper function to preprocess the image so that it can be inputted in MobileNetV2
    def preprocess(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = image[None, ...]
        return image

    # Helper function to extract labels from probability vector
    def get_imagenet_label(probs):
        return decode_predictions(probs, top=1)[0][0]

    # file_name = '800px-Sunflower_sky_backdrop.jpg'
    # file_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg'
    # 여기에 이미지 주소 입력받아 넣을거임
    image_path = tf.keras.utils.get_file(file_name, file_path)
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)

    image = preprocess(image)
    image_probs = pretrained_model.predict(image)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = pretrained_model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    # perturbations = create_adversarial_pattern(image, label)

    # plt.imshow(perturbations[0]*0.5+0.5); # To change [-1, 1] to [0,1]

    def display_images(image, description):
        _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
        plt.figure()
        plt.imshow(image[0] * 0.5 + 0.5)
        plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                         label, confidence * 100))
        # plt.show()
        img_path = file_name+".png"
        plt.savefig("static/" + img_path)
        return img_path

    # epsilons = [0, 0.001, 0.005, 0.01, 0.1]
    #epsilons = 0.1
    descriptions = "Epsilon = " + str(epsilon)

    # descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
    #                for eps in epsilons]

    def adv_Bim(eps):
        N = 10
        adv_x = image
        for i in range(N):
            perturbations = create_adversarial_pattern(adv_x, label)
            adv_x = adv_x + eps * perturbations
            adv_x = tf.clip_by_value(adv_x, -1, 1)
        return adv_x

    # for i, eps in enumerate(epsilons):
    #    adv_x = adv_Bim(eps)
    #    display_images(adv_x, descriptions[i])
    def get_image():
        adv_x = adv_Bim(epsilon)
        img_path = display_images(adv_x, descriptions)
        pred_all, distances_all = detector_all.test(adv_x)
        pred_bit, distances_bit = detector_bit.test(adv_x)
        pred_median, distances_median = detector_median.test(adv_x)
        pred_nlocal, distances_nlocal = detector_nlocal.test(adv_x)

        print('all squeezer = distance: %f, detection: %s' %(distances_all, pred_all))
        print('bit depth = distance: %f, detection: %s' % (distances_bit, pred_bit))
        print('median = distance: %f, detection: %s' % (distances_median, pred_median))
        print('non local = distance: %f, detection: %s' % (distances_nlocal, pred_nlocal))

        return img_path, pred_all

    return get_image()


