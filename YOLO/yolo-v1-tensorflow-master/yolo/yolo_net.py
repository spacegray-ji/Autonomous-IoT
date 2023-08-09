        self.classes = cfg.CLASSES  # 类别
        self.num_class = len(self.classes)  # 类别数量20
        self.image_size = cfg.IMAGE_SIZE  # 图像尺寸448
        self.cell_size = cfg.CELL_SIZE  # cell尺寸7
        self.boxes_per_cell = cfg.BOXES_PER_CELL  # 每个grid cell负责的box数量，文论中为2
        self.output_size = (self.cell_size * self.cell_size) * \
                           (self.num_class + self.boxes_per_cell * 5)  # 输出尺寸(7*7)*(20+2*5)
        self.scale = 1.0 * self.image_size / self.cell_size  # 图像尺寸和cell尺寸比例
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  # 7*7*20
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell  # 7*7*20 + 7*7*2

        self.object_scale = cfg.OBJECT_SCALE  # 值为1.0
        self.noobject_scale = cfg.NOOBJECT_SCALE  # 值为1.0
        self.class_scale = cfg.CLASS_SCALE  # 值为2.0
        self.coord_scale = cfg.COORD_SCALE  # 值为5.0

        self.learning_rate = cfg.LEARNING_RATE  # 学习速率LEARNING_RATE = 0.0001
        self.batch_size = cfg.BATCH_SIZE  # BATCH_SIZE = 45
        self.alpha = cfg.ALPHA  # ALPHA = 0.1

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))  # 偏置 形状为 (7,7,2)

        # self.images = tf.placeholder(
        #     tf.float32, [None, self.image_size, self.image_size, 3],
        #     name='images')  # 输入图片 (None,448,448,3)

        self.images = tf.Variable(tf.ones(shape=[None, self.image_size, self.image_size, 3]), dtype=tf.float32, name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)  # 构建网络 获取网络输出 形状为 (None,1470)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])  # 标签 (None,7,7,25)
            self.loss_layer(self.logits, self.labels)  # 损失函数
            self.total_loss = tf.losses.get_total_loss()  # 加入权重之后的损失函数
            tf.summary.scalar('total_loss', self.total_loss)  # 将损失以标量形式显示 命名为total_loss

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        """
            构建YOLO网络

            args:
                images：输入图片占位符 [None,image_size,image_size,3]  这里是[None,448,448,3]
                num_outputs：标量，网络输出节点数 1470
                alpha：泄露修正线性激活函数 系数0.1
                keep_prob：弃权 保留率
                is_training：训练？
                scope：命名空间名

            return：
                返回网络最后一层，激活函数处理之前的值  形状[None,1470]
        """
        with tf.variable_scope(scope):  # 定义变量命名空间
            with keras.layers.arg_scope(  # 定义共享参数 使用L2正则化
                    [keras.layers.conv2d, keras.layers.fully_connected],
                    activation_fn=leaky_relu(alpha),
                    weights_regularizer=keras.layers.l2_regularizer(0.0005),
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')  # pad_1填充成 (None,454,454,3)
                # TODO 替换slim成keras
                net = keras.layers.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = keras.layers.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = keras.layers.conv2d(net, 192, 3, scope='conv_4')
                net = keras.layers.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = keras.layers.conv2d(net, 128, 1, scope='conv_6')
                net = keras.layers.conv2d(net, 256, 3, scope='conv_7')
                net = keras.layers.conv2d(net, 256, 1, scope='conv_8')
                net = keras.layers.conv2d(net, 512, 3, scope='conv_9')
                net = keras.layers.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = keras.layers.conv2d(net, 256, 1, scope='conv_11')
                net = keras.layers.conv2d(net, 512, 3, scope='conv_12')
                net = keras.layers.conv2d(net, 256, 1, scope='conv_13')
                net = keras.layers.conv2d(net, 512, 3, scope='conv_14')
                net = keras.layers.conv2d(net, 256, 1, scope='conv_15')
                net = keras.layers.conv2d(net, 512, 3, scope='conv_16')
                net = keras.layers.conv2d(net, 256, 1, scope='conv_17')
                net = keras.layers.conv2d(net, 512, 3, scope='conv_18')
                net = keras.layers.conv2d(net, 512, 1, scope='conv_19')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_20')
                net = keras.layers.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = keras.layers.conv2d(net, 512, 1, scope='conv_22')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_23')
                net = keras.layers.conv2d(net, 512, 1, scope='conv_24')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_25')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = keras.layers.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_29')
                net = keras.layers.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = keras.layers.flatten(net, scope='flat_32')
                net = keras.layers.fully_connected(net, 512, scope='fc_33')
                net = keras.layers.fully_connected(net, 4096, scope='fc_34')
                net = keras.layers.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = keras.layers.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        这个函数的主要作用是计算两个 bounding box 之间的 IoU。输入是两个 5 维的bounding box,输出的两个 bounding Box 的IoU
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
          注意这里的参数x_center, y_center, w, h都是归一到[0,1]之间的，分别表示预测边界框的中心相对整张图片的坐标，宽和高
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            # 把以前的中心点坐标和长和宽转换成了左上角和右下角的两个点的坐标
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            # 注意此坐标系 坐标原点 为左上角！！！
            # lu和rd就是分别求两个框相交的矩形的左上角的坐标和右下角的坐标，因为对于左上角，
            # 选择的是x和y较大的，而右下角是选择较小的，可以想想两个矩形框相交是不是这中情况
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])  # lu(left up) 两个框相交的矩形的左上角(x1,y1)
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])  # rd(right down) 两个框相交的矩形的右下角(x2,y2)

            # intersection
            # 这个就是求相交矩形的长和宽，所以有rd-lu，相当于x1-x2和y1-y2，
            # 之所以外面还要加一个tf.maximum是因为删除那些不合理的框，比如两个框没交集，
            # 就会出现左上角坐标比右下角还大。
            intersection = tf.maximum(0.0, rd - lu)
            # inter_square这个就是求面积了，就是长乘以宽。
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            # square1和square2这个就是求面积了，因为之前是中心点坐标和长和宽，所以这里直接用长和宽
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            # union_square就是就两个框的交面积，因为如果两个框的面积相加，那就会重复了相交的部分，
            # 所以减去相交的部分，外面有个tf.maximum这个就是保证相交面积不为0,因为后面要做分母。
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        # 最后有一个tf.clip_by_value,这个是将如果你的交并比大于1,那么就让它等于1,如果小于0,那么就
        # 让他变为0,因为交并比在0-1之间。
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        """
        计算预测和标签之间的损失函数
            args：
                predicts：Yolo网络的输出 形状[None,1470]
                0：7*7*20：表示预测类别
                7*7*20:7*7*20 + 7*7*2:表示预测置信度，即预测的边界框与实际边界框之间的IOU
                7*7*20 + 7*7*2：1470：预测边界框    目标中心是相对于当前格子的，宽度和高度的开根号是相对当前整张图像的(归一化的)
                labels：标签值 形状[None,7,7,25]
                0:1：置信度，表示这个地方是否有目标
                1:5：目标边界框  目标中心，宽度和高度(没有归一化)
                5:25：目标的类别
        """
        with tf.variable_scope(scope):
            # 将网络输出分离为类别和置信度以及边界框的大小，输出维度为7*7*20 + 7*7*2 + 7*7*2*4=1470
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])  # 预测每个格子目标的类别 形状[45,7,7,20]
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])  # 预测每个格子中两个边界框的置信度 形状[45,7,7,2]
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            # 预测每个格子中的两个边界框，(x,y)表示边界框相对于格子边界框的中心 w,h的开根号相对于整个图片  形状[45,7,7,2,4]
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])  # 标签的置信度,表示这个地方是否有框 形状[45,7,7,1]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])  # 标签的边界框 (x,y)表示边界框相对于整个图片的中心 形状[45,7,7,1，4]
            boxes = tf.tile(
                boxes,
                [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size  # 标签的边界框 归一化后 张量沿着axis=3重复两边，扩充后[45,7,7,2,4]
            classes = labels[..., 5:]

            """
            predict_boxes_tran：offset变量用于把预测边界框predict_boxes中的坐标中心(x,y)由相对当前格子转换为相对当前整个图片
            offset，这个是构造的[7,7,2]矩阵，每一行都是[7,2]的矩阵，值为[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
            这个变量是为了将每个cell的坐标对齐，后一个框比前一个框要多加1
            比如我们预测了cell_size的每个中心点坐标，那么我们这个中心点落在第几个cell_size
            就对应坐标要加几，这个用法比较巧妙，构造了这样一个数组，让他们对应位置相加
            """
            # offset shape为[1,7,7,2]  如果忽略axis=0，则每一行都是  [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            ''
            # shape为[45,7,7,2]  如果忽略axis=0 第i行为[[i,i],[i,i],[i,i],[i,i],[i,i],[i,i],[i,i]]
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))

            # shape为[45,7,7,2,4]  计算每个格子中的预测边界框坐标(x,y)相对于整个图片的位置  而不是相对当前格子
            # 假设当前格子为(3,3)，当前格子的预测边界框为(x0,y0)，则计算坐标(x,y) = ((x0,y0)+(3,3))/7
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            # 计算每个格子预测边界框与真实边界框之间的IOU  [45,7,7,2]
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # 这个是求论文中的1ijobj参数，[45,7,7,2]     1ijobj：表示网格单元i的第j个编辑框预测器’负责‘该预测
            # 先计算每个框交并比最大的那个，因为我们知道，YOLO每个格子预测两个边界框，一个类别。在训练时，每个目标只需要
            # 一个预测器来负责，我们指定一个预测器"负责"，根据哪个预测器与真实值之间具有当前最高的IOU来预测目标。
            # 所以object_mask就表示每个格子中的哪个边界框负责该格子中目标预测？哪个边界框取值为1，哪个边界框就负责目标预测
            # 当格子中的确有目标时，取值为[1,1]，[1,0],[0,1]
            # 比如某一个格子的值为[1,0]，表示第一个边界框负责该格子目标的预测  [0,1]：表示第二个边界框负责该格子目标的预测
            # 当格子没有目标时，取值为[0,0]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # noobject_mask就表示每个边界框不负责该目标的置信度，
            # 使用tf.onr_like，使得全部为1,再减去有目标的，也就是有目标的对应坐标为1,这样一减，就变为没有的了。[45,7,7,2]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            # boxes_tran 这个就是把之前的坐标换回来(相对整个图像->相对当前格子)，长和宽开方(原因在论文中有说明)，后面求loss就方便。 shape为(4, 45, 7, 7, 2)
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss 分类损失，如果目标出现在网格中 response为1，否则response为0  原文代价函数公式第5项
            # 该项表名当格子中有目标时，预测的类别越接近实际类别，代价值越小  原文代价函数公式第5项
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss 有目标物体存在的置信度预测损失   原文代价函数公式第3项
            # 该项表名当格子中有目标时，负责该目标预测的边界框的置信度越越接近预测的边界框与实际边界框之间的IOU时，代价值越小
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss  没有目标物体存在的置信度的损失(此时iou_predict_truth为0)  原文代价函数公式第4项
            # 该项表名当格子中没有目标时，预测的两个边界框的置信度越接近0，代价值越小
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss 边界框坐标损失 shape 为 [batch_size, 7, 7, 2, 1]  原文代价函数公式1,2项
            # 该项表名当格子中有目标时，预测的边界框越接近实际边界框，代价值越小
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            # 将所有损失放在一起
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # 将每个损失添加到日志记录
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    return op