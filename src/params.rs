use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 按名读取张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).expect(&format!("Failed to find tensor with name: {}", name));
            let shape = tensor_view.shape().to_vec();
            let data = tensor_view.data();
            let data: Vec<f32> = data
                .chunks_exact(4)// 将字节数据按4字节（f32 的大小）分块
                .filter_map(|chunk|{
                    // 确保每个块正好是4字节，并将其转成f32，若不是则跳过
                    chunk.try_into().ok().map(|array:[u8;4]| f32::from_le_bytes(array))
                })
                .collect();
            Tensor::<f32>::new(data, &shape)
        };
        
        // 实现对每层的按名读取
        let get_layer_tensors = |suffix: &str| -> Vec<Tensor<f32>> {
            (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}",i,suffix)))
                .collect()
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_layer_tensors("input_layernorm.weight"),
            wq: get_layer_tensors("self_attn.q_proj.weight"),
            wk: get_layer_tensors("self_attn.k_proj.weight"),
            wv: get_layer_tensors("self_attn.v_proj.weight"),
            wo: get_layer_tensors("self_attn.o_proj.weight"),
            rms_ffn_w: get_layer_tensors("post_attention_layernorm.weight"),
            w_up: get_layer_tensors("mlp.up_proj.weight"),
            w_gate: get_layer_tensors("mlp.gate_proj.weight"),
            w_down: get_layer_tensors("mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
