function convertToTensor(image) {
    var contentTensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims(0);

    return contentTensor
}

function getImageTensors() {
    var contentImage = document.getElementById("content-image-selector")
    var styleImage = document.getElementById("style-image-selector")

    var contentTensor = convertToTensor(contentImage)
    var styleTensor = convertToTensor(styleImage)

    return {
        contentTensor,
        styleTensor
    }
}


function myFunction() {
    var images = getImageTensors()
    var contentTensor = images.contentTensor
    var styleTensor = images.styleTensor

    var modelPromise = tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/resnet_v2_50/feature_vector/3/default/1", {
        fromTFHub: true
    })

    modelPromise.then(async function (model) {
        var contentFeatures = getFeatures(contentTensor, model)
        var styleFeatures = getFeatures(styleTensor, model)

        var styleGrams = {}
        // calculate the gram matrices for each layer of out style represetation
        for (var key in styleFeatures) {
            var features = styleFeatures[key]
            var gramMatrix = computeGramMatrix(features)
            styleGrams[key] = gramMatrix
        }

        const contentWeight = tf.scalar(1.)
        const styleWeight = tf.scalar(1000000.)

        const target = tf.variable(contentTensor)

        const optimizer = tf.train.adam(0.1)

        var styleWeights = {
            'module_apply_default/resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/Relu': 1.,
            'module_apply_default/resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/Relu': 0.8,
            'module_apply_default/resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/Relu': 0.5,
            'module_apply_default/resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/Relu': 0.3
        }

        async function train_one_epoch() {
            optimizer.minimize(() => {

                // get tje featires from the target image 
                var targetFeatures = getFeatures(target, model)

                // compute the content loss
                // var contentLoss = tf.mean(tf.pow(tf.sub(targetFeatures[3], contentFeatures[3]), 2))
                var contentLoss = tf.losses.meanSquaredError(targetFeatures['module_apply_default/resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/Relu'],
                    contentFeatures['module_apply_default/resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/Relu'])

                var styleLoss = 0

                for (var layer in styleWeights) {

                    // get the target 'style' representations for the layer
                    var targetFeature = targetFeatures[layer]

                    const shape = targetFeature.shape

                    // calculate the gram matrix for the target representation
                    const targetGram = computeGramMatrix(targetFeature)

                    // get the gra matrix fro style representation
                    const styleGram = styleGrams[layer]

                    const layerStyleLoss = tf.mul(styleWeights[layer], tf.losses.meanSquaredError(targetGram, styleGram))

                    // add to the style loss
                    styleLoss = tf.add(styleLoss, tf.div(layerStyleLoss, (shape[3] * shape[1] * shape[2])))
                }

                const totalLoss = tf.add(tf.mul(contentWeight, contentLoss), tf.mul(styleWeight, styleLoss))

                //console.log('Epoch', epoch, "Loss:", totalLoss.print());

                return totalLoss
            });
        }


        for (let epoch = 0; epoch < 500; epoch++) {
            await train_one_epoch()
        }

        updateImage(target)

    })
}

function updateImage(imageTensor) {
    var shape = imageTensor.shape

    imageTensor.data().then(function (image) {
        var ctx = document.getElementById('imageCanvas').getContext('2d');

        var count = 0

        for (row = 0; row < shape[1]; row++) {
            for (col = 0; col < shape[2]; col++) {
                ctx.fillStyle = `rgb(${image[count]}, ${image[count+1]}, ${image[count+2]})`;
                ctx.fillRect(col, row, 1, 1);
                count += 3
            }
        }
    })
}

function computeGramMatrix(features) {
    const shape = features.shape

    features = features.reshape([shape[0], shape[1] * shape[2], shape[3]])
    features = features.squeeze()
    features = tf.transpose(features, [1, 0])
    var matrix = tf.matMul(features, features.transpose())
    return matrix
}

function getFeatures(imageTensor, model) {
    const layers = ['module_apply_default/resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/Relu',
                 'module_apply_default/resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/Relu',
                 'module_apply_default/resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/Relu',
                 'module_apply_default/resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/Relu']

    var features = model.execute(imageTensor, layers)

    var result = {};
    layers.forEach((key, i) => result[key] = features[i]);

    return result
}
