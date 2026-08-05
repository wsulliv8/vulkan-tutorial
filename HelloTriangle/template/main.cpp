#include <memory>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <chrono>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  std::vector<char> buffer(file.tellg());
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  file.close();
  return buffer;
}

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  static std::array<vk::VertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    return {{{.location = 0,
              .binding = 0,
              .format = vk::Format::eR32G32B32Sfloat,
              .offset = offsetof(Vertex, pos)},
             {.location = 1,
              .binding = 0,
              .format = vk::Format::eR32G32B32Sfloat,
              .offset = offsetof(Vertex, color)},
             {.location = 2,
              .binding = 0,
              .format = vk::Format::eR32G32Sfloat,
              .offset = offsetof(Vertex, texCoord)}}};
  }

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord;
  }
};

namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
} // namespace std

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;
  bool framebufferResized = false;

  std::vector<vk::Image> swapChainImages;
  std::vector<vk::raii::ImageView> swapChainImageViews;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;

  vk::Extent2D swapChainExtent;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  vk::raii::Pipeline graphicsPipeline = nullptr;

  std::vector<Vertex> vertices;
  std::unordered_map<Vertex, uint32_t> uniqueVertices;
  std::vector<uint32_t> indices;
  vk::raii::Buffer vertexBuffer = nullptr;
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;

  vk::raii::Buffer indexBuffer = nullptr;
  vk::raii::DeviceMemory indexBufferMemory = nullptr;

  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  vk::raii::Buffer stagingBuffer = nullptr;
  vk::raii::DeviceMemory stagingBufferMemory = nullptr;

  uint32_t mipLevels = 0;
  vk::raii::Image textureImage = nullptr;
  vk::raii::DeviceMemory textureImageMemory = nullptr;
  vk::raii::ImageView textureImageView = nullptr;
  vk::raii::Sampler textureSampler = nullptr;

  vk::raii::Image depthImage = nullptr;
  vk::raii::DeviceMemory depthImageMemory = nullptr;
  vk::raii::ImageView depthImageView = nullptr;

  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t frameIndex = 0;

  vk::raii::Queue queue = nullptr;
  uint32_t queueIndex = ~0;

  std::vector<const char *> requiredDeviceExtension = {
      vk::KHRSwapchainExtensionName};

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffer();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    device.waitIdle();
  }

  void createInstance() {
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14,
    };

    // Get the required layers
    std::vector<char const *> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan
    // implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
        requiredLayers, [&layerProperties](auto const &requiredLayer) {
          return std::ranges::none_of(
              layerProperties, [requiredLayer](auto const &layerProperty) {
                return std::strcmp(layerProperty.layerName, requiredLayer) == 0;
              });
        });
    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(*unsupportedLayerIt));
    }

    // Get the required extensions.

    auto requiredExtensions = getRequiredInstanceExtensions();

    auto extensionProperties = context.enumerateInstanceExtensionProperties();

    std::cout << "Available extensions:" << std::endl;
    for (const auto &extension : extensionProperties) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }

    // Check if the required extensions are supported by the Vulkan
    // implementation.
    auto unsupportedPropertyIt = std::ranges::find_if(
        requiredExtensions,
        [&extensionProperties](auto const &requiredExtension) {
          return std::ranges::none_of(
              extensionProperties,
              [requiredExtension](auto const &extensionProperty) {
                return std::strcmp(extensionProperty.extensionName,
                                   requiredExtension) == 0;
              });
        });
    if (unsupportedPropertyIt != requiredExtensions.end()) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data(),
    };

    instance = vk::raii::Instance(context, createInfo);
  }

  std::vector<const char *> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    return extensions;
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  void pickPhysicalDevice() {
    auto physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
      throw std::runtime_error("Failed to find a suitable physical device");
    }

    // use an ordered map to automatically sort candidates by increasing score
    std::multimap<int, vk::raii::PhysicalDevice> candidates;
    for (const auto &pd : physicalDevices) {
      auto deviceProperties = pd.getProperties();
      auto deviceFeatures = pd.getFeatures();

      uint32_t score = 0;
      // Discrete GPU
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score = 1000;
      }

      // max possible size of textures affects graphics quality
      score += deviceProperties.limits.maxImageDimension2D;

      if (!deviceFeatures.geometryShader) {
        continue;
      }

      if (!isDeviceSuitable(pd)) {
        continue;
      }

      candidates.insert(std::make_pair(score, pd));
    }

    if (!candidates.empty() && candidates.rbegin()->first > 0) {
      physicalDevice = candidates.rbegin()->second;
    } else {
      throw std::runtime_error("Failed to find a suitable physical device");
    }
  }

  bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice) {
    auto deviceProperties = physicalDevice.getProperties();
    auto deviceFeatures = physicalDevice.getFeatures();

    // Check if the physicalDevice supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 = deviceProperties.apiVersion >= vk::ApiVersion13;

    // Check if any of the queue families support graphics operations
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    // Check if all required physicalDevice extensions are available
    auto availableDeviceExtensions =
        physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        requiredDeviceExtension,
        [&availableDeviceExtensions](auto const &requiredExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions,
              [requiredExtension](auto const &availableDeviceExtension) {
                return std::strcmp(availableDeviceExtension.extensionName,
                                   requiredExtension) == 0;
              });
        });

    // Check if the physicalDevice supports the required features
    auto features = physicalDevice.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceFeatures2>()
            .features.samplerAnisotropy &&
        features.template get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphics &&
           supportsAllRequiredExtensions && supportsRequiredFeatures;
  }

  void createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both
    // graphics and present
    queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
         qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags &
           vk::QueueFlagBits::eGraphics) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        // found a queue family that supports both graphics and present
        queueIndex = qfpIndex;
        break;
      }
    }
    if (queueIndex == ~0) {
      throw std::runtime_error(
          "Could not find a queue for graphics and present -> terminating");
    }

    // query for Vulkan 1.3 features
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        featureChain = {
            {.features = {.samplerAnisotropy =
                              true}}, // vk::PhysicalDeviceFeatures2
            {.shaderDrawParameters =
                 true}, // vk::PhysicalDeviceVulkan11Features
            {.synchronization2 = true,
             .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
            {.extendedDynamicState =
                 true}, // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        };

    // create a Device
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
        .queueFamilyIndex = queueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredDeviceExtension.size()),
        .ppEnabledExtensionNames = requiredDeviceExtension.data(),
    };

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);

    queue = vk::raii::Queue(device, queueIndex, 0);
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      std::vector<vk::SurfaceFormatKHR> const &availableFormats) {
    const auto formatIt =
        std::ranges::find_if(availableFormats, [](const auto &format) {
          return format.format == vk::Format::eB8G8R8A8Srgb &&
                 format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(
      std::vector<vk::PresentModeKHR> const &availablePresentModes) {
    assert(
        std::ranges::any_of(availablePresentModes, [](const auto &presentMode) {
          return presentMode == vk::PresentModeKHR::eFifo;
        }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) {
                                 return value == vk::PresentModeKHR::eMailbox;
                               })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D
  chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                                 capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                                 capabilities.maxImageExtent.height)};
  }

  uint32_t chooseSwapMinImageCount(
      vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) &&
        (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  void createSwapChain() {
    auto surfaceCapabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats =
        physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
    std::vector<vk::PresentModeKHR> availablePresentModes =
        physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *surface,
        .minImageCount = minImageCount,
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
  }

  void recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    device.waitIdle();
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
  }

  void cleanupSwapChain() {
    swapChainImages.clear();
    swapChain = nullptr;
  }

  vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format,
                                      vk::ImageAspectFlags aspectFlags,
                                      uint32_t mipLevels) {
    vk::ImageViewCreateInfo viewInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange =
            {
                .aspectMask = aspectFlags,
                .levelCount = mipLevels,
                .layerCount = 1,
            },
    };
    return vk::raii::ImageView(device, viewInfo);
  }

  void createImageViews() {
    assert(!swapChainImages.empty());
    swapChainImageViews.reserve(swapChainImages.size());
    for (auto &image : swapChainImages) {
      swapChainImageViews.emplace_back(
          createImageView(image, swapChainSurfaceFormat.format,
                          vk::ImageAspectFlagBits::eColor, 1));
    }
  }

  void createSyncObjects() {
    assert(presentCompleteSemaphores.empty() &&
           renderFinishedSemaphores.empty() && inFlightFences.empty());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
      inFlightFences.emplace_back(
          device, vk::FenceCreateInfo{
                      .flags = vk::FenceCreateFlagBits::eSignaled,
                  });
    }
  }

  void cleanup() {
    cleanupSwapChain();
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  [[nodiscard]] vk::raii::ShaderModule
  createShaderModule(const std::vector<char> &code) const {
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };
    return vk::raii::ShaderModule(device, createInfo);
  }

  void createDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
        {{.binding = 0,
          .descriptorType = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eVertex},
         {.binding = 1,
          .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eFragment}}};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  }

  void createGraphicsPipeline() {
    auto shaderCode = readFile("shaders/slang.spv");
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain",
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain",
    };
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                        fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
    };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vk::Rect2D scissor{
        .offset = vk::Offset2D{0, 0},
        .extent = swapChainExtent,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f,
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False,
    };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*descriptorSetLayout,
        .pushConstantRangeCount = 0,
    };
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::Format depthFormat = findDepthFormat();

    vk::StructureChain<vk::GraphicsPipelineCreateInfo,
                       vk::PipelineRenderingCreateInfo>
        pipelineCreateInfoChain = {
            {.stageCount = 2,
             .pStages = shaderStages,
             .pVertexInputState = &vertexInputInfo,
             .pInputAssemblyState = &inputAssembly,
             .pViewportState = &viewportState,
             .pRasterizationState = &rasterizer,
             .pMultisampleState = &multisampling,
             .pDepthStencilState = &depthStencil,
             .pColorBlendState = &colorBlending,
             .pDynamicState = &dynamicState,
             .layout = pipelineLayout,
             .renderPass = nullptr},
            {.colorAttachmentCount = 1,
             .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
             .depthAttachmentFormat = depthFormat}};

    graphicsPipeline = vk::raii::Pipeline(
        device, nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    vk::raii::CommandBuffer commandBuffer =
        std::move(device.allocateCommandBuffers(allocInfo).front());
    vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    commandBuffer.begin(beginInfo);
    return std::move(commandBuffer);
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer) {
    commandBuffer.end();
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer,
    };
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex,
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties =
        physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error("Failed to find suitable memory type");
  }

  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) {
    vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties),
    };
    vk::raii::DeviceMemory bufferMemory =
        vk::raii::DeviceMemory(device, memoryAllocateInfo);
    buffer.bindMemory(*bufferMemory, 0);
    return {std::move(buffer), std::move(bufferMemory)};
  }

  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size) {
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer,
                                 vk::BufferCopy(0, 0, size));
    endSingleTimeCommands(std::move(commandCopyBuffer));
  }

  void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer,
                         const vk::raii::Buffer &buffer,
                         const vk::raii::Image &image, uint32_t width,
                         uint32_t height) {
    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };
    commandBuffer.copyBufferToImage(
        *buffer, *image, vk::ImageLayout::eTransferDstOptimal, region);
  }

  void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer,
                             const vk::raii::Image &image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier{
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .layerCount = 1,
        }};
    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("Unsupported layout transition!");
    }
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {},
                                  barrier);
  }

  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
              vk::Format format, vk::ImageTiling tiling,
              vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties) {
    vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    vk::raii::Image image = vk::raii::Image(device, imageInfo);
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties),
    };
    vk::raii::DeviceMemory imageMemory =
        vk::raii::DeviceMemory(device, memoryAllocateInfo);
    image.bindMemory(*imageMemory, 0);
    return {std::move(image), std::move(imageMemory)};
  }

  void generateMipmaps(vk::raii::CommandBuffer &commandBuffer,
                       const vk::raii::Image &image, vk::Format imageFormat,
                       int32_t texWidth, int32_t texHeight,
                       uint32_t mipLevels) {
    vk::FormatProperties formatProperties =
        physicalDevice.getFormatProperties(imageFormat);
    if (!(formatProperties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
      throw std::runtime_error(
          "Texture image format does not support linear blitting!");
    }
    vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
    };
    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;
    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eTransfer, {},
                                    {}, {}, barrier);
      vk::ImageBlit blit{
          .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = i - 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
          .srcOffsets = std::array<vk::Offset3D, 2>(
              {{0, 0, 0}, {mipWidth, mipHeight, 1}}),
          .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = i,
                             .layerCount = 1},
          .dstOffsets = std::array<vk::Offset3D, 2>(
              {{},
               {1 < mipWidth ? mipWidth / 2 : 1,
                1 < mipHeight ? mipHeight / 2 : 1, 1}})};
      commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                              image, vk::ImageLayout::eTransferDstOptimal, blit,
                              vk::Filter::eLinear);
      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eFragmentShader,
                                    {}, {}, {}, barrier);
      if (mipWidth > 1)
        mipWidth /= 2;
      if (mipHeight > 1)
        mipHeight /= 2;
    }
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  {}, {}, {}, barrier);
  }

  void createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);
    mipLevels = static_cast<uint32_t>(
                    std::floor(std::log2(std::max(texWidth, texHeight)))) +
                1;
    vk::DeviceSize imageSize = texWidth * texHeight * 4;
    if (!pixels) {
      throw std::runtime_error("Failed to load texture image!");
    }
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();
    stbi_image_free(pixels);
    std::tie(textureImage, textureImageMemory) =
        createImage(texWidth, texHeight, mipLevels, vk::Format::eR8G8B8A8Srgb,
                    vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eTransferSrc |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    transitionImageLayout(commandBuffer, textureImage,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(commandBuffer, stagingBuffer, textureImage,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    generateMipmaps(commandBuffer, textureImage, vk::Format::eR8G8B8A8Srgb,
                    texWidth, texHeight, mipLevels);
    endSingleTimeCommands(std::move(commandBuffer));
  }

  void createTextureImageView() {
    textureImageView =
        createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageAspectFlagBits::eColor, mipLevels);
  }

  void createTextureSampler() {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = VK_FALSE,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = vk::LodClampNone,
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = VK_FALSE,
    };
    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (const auto format : candidates) {
      vk::FormatProperties props = physicalDevice.getFormatProperties(format);
      if ((tiling == vk::ImageTiling::eLinear &&
           (props.linearTilingFeatures & features) == features) ||
          (tiling == vk::ImageTiling::eOptimal &&
           (props.optimalTilingFeatures & features) == features)) {
        return format;
      }
    }
    throw std::runtime_error("Failed to find supported format!");
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    std::tie(depthImage, depthImageMemory) =
        createImage(swapChainExtent.width, swapChainExtent.height, 1,
                    depthFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageView = createImageView(depthImage, depthFormat,
                                     vk::ImageAspectFlagBits::eDepth, 1);
  }

  void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          MODEL_PATH.c_str())) {
      throw std::runtime_error(warn + err);
    }
    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        Vertex vertex{};
        vertex.pos = {
            attrib.vertices[3 * index.vertex_index + 0],
            attrib.vertices[3 * index.vertex_index + 1],
            attrib.vertices[3 * index.vertex_index + 2],
        };
        vertex.color = {1.0f, 1.0f, 1.0f};
        vertex.texCoord = {
            attrib.texcoords[2 * index.texcoord_index + 0],
            1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
        };
        auto [it, inserted] = uniqueVertices.insert(
            {vertex, static_cast<uint32_t>(vertices.size())});
        if (inserted) {
          vertices.push_back(vertex);
        }
        indices.push_back(it->second);
      }
    }
  }

  void createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(vertexBuffer, vertexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }

  void createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(indexBuffer, indexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }

  void createUniformBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      auto [buffer, bufferMem] =
          createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);
      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMem));
      uniformBuffersMapped.emplace_back(
          uniformBuffersMemory.back().mapMemory(0, bufferSize));
    }
  }

  void createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{{
        {.type = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = MAX_FRAMES_IN_FLIGHT},
        {.type = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = MAX_FRAMES_IN_FLIGHT},
    }};
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
  }

  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data(),
    };
    descriptorSets = device.allocateDescriptorSets(allocInfo);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo bufferInfo{
          .buffer = uniformBuffers[i],
          .offset = 0,
          .range = sizeof(UniformBufferObject),
      };
      vk::DescriptorImageInfo imageInfo{
          .sampler = textureSampler,
          .imageView = textureImageView,
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      };
      std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
          {{
               .dstSet = descriptorSets[i],
               .dstBinding = 0,
               .dstArrayElement = 0,
               .descriptorCount = 1,
               .descriptorType = vk::DescriptorType::eUniformBuffer,
               .pBufferInfo = &bufferInfo,
           },
           {
               .dstSet = descriptorSets[i],
               .dstBinding = 1,
               .dstArrayElement = 0,
               .descriptorCount = 1,
               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
               .pImageInfo = &imageInfo,
           }}};
      device.updateDescriptorSets(descriptorWrites, {});
    }
  }

  void createCommandBuffer() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }

  void transition_image_layout(vk::Image image, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask,
                               vk::ImageAspectFlags image_aspect_flags) {
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = src_stage_mask,
        .srcAccessMask = src_access_mask,
        .dstStageMask = dst_stage_mask,
        .dstAccessMask = dst_access_mask,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = image_aspect_flags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});

    // Before starting rendering, transition the swapchain image to
    // vk::ImageLayout::eColorAttachmentOptimal
    transition_image_layout(
        swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
        vk::ImageAspectFlagBits::eColor);
    transition_image_layout(
        *depthImage, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
            vk::PipelineStageFlagBits2::eLateFragmentTests, // srcStage
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
            vk::PipelineStageFlagBits2::eLateFragmentTests, // dstStage
        vk::ImageAspectFlagBits::eDepth);
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::RenderingAttachmentInfo colorAttachInfo = {
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor,
    };
    vk::RenderingAttachmentInfo depthAttachInfo = {
        .imageView = depthImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth,
    };
    vk::RenderingInfo renderingInfo = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = swapChainExtent,
            },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachInfo,
        .pDepthAttachment = &depthAttachInfo,
    };
    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               *graphicsPipeline);
    commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffer.bindIndexBuffer(
        *indexBuffer, 0,
        vk::IndexTypeValue<decltype(indices)::value_type>::value);
    commandBuffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                     static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0,
                             vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     pipelineLayout, 0,
                                     *descriptorSets[frameIndex], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,
                              0);
    commandBuffer.endRendering();

    // After rendering, transition the swapchain image to
    // vk::ImageLayout::ePresentSrcKHR
    transition_image_layout(
        swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
        {},                                                 // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe,          // dstStage
        vk::ImageAspectFlagBits::eColor);
    commandBuffer.end();
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();
    UniformBufferObject ubo = {};
    // ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
    //                         glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.model = glm::mat4(1.0f);
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                static_cast<float>(swapChainExtent.width) /
                                    static_cast<float>(swapChainExtent.height),
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1.0f;
    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }

  void drawFrame() {
    auto fenceResult =
        device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for fence");
    }
    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, presentCompleteSemaphores[frameIndex], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    }
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("Failed to acquire swapchain image");
    }

    updateUniformBuffer(frameIndex);

    device.resetFences(*inFlightFences[frameIndex]);

    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex],
    };
    queue.submit(submitInfo, inFlightFences[frameIndex]);

    const vk::PresentInfoKHR presentInfoKHR = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex,
    };
    result = queue.presentKHR(presentInfoKHR);
    if ((result == vk::Result::eSuboptimalKHR) ||
        (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else {
      assert(result == vk::Result::eSuccess);
    }
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }
};

int main() {
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
